import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)          # fixed seed so results repeat
results = {}

# ---------------------------------------------------------------------------
# STEP 1: make the dataset (in the real project this is replaced by read_csv)
# ---------------------------------------------------------------------------
N_STORES = 20
stores = pd.DataFrame({
    "store_id": range(1, N_STORES + 1),
    "x_km": rng.uniform(-20, 20, N_STORES).round(1),
    "y_km": rng.uniform(-20, 20, N_STORES).round(1),
    "size_factor": rng.choice([0.6, 1.0, 1.5], N_STORES, p=[0.3, 0.45, 0.25]),
})
# warehouse sits at (0, 0). Road distance is a bit more than straight line.
ROAD_FACTOR = 1.3
stores["distance_km"] = (np.hypot(stores.x_km, stores.y_km) * ROAD_FACTOR).round(1)

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
festival_windows = [("2025-06-14", "2025-06-16"), ("2025-09-28", "2025-10-02"),
                    ("2025-10-18", "2025-10-22"), ("2025-12-24", "2025-12-31")]
festival = np.zeros(len(dates), dtype=int)
for a, b in festival_windows:
    festival[(dates >= a) & (dates <= b)] = 1

rows = []
for i, d in enumerate(dates):
    dow = d.dayofweek
    weekend_boost = 1.25 if dow in (5, 6) else 1.0
    trend = 1 + 0.0004 * i                      # business slowly growing
    fest_boost = 1.35 if festival[i] else 1.0
    for _, s in stores.iterrows():
        mean_qty = 70 * s.size_factor * weekend_boost * trend * fest_boost
        qty = max(rng.normal(mean_qty, mean_qty * 0.18), 5)
        trip_time = s.distance_km / 22 + 0.5 + rng.exponential(0.9)
        rows.append([d, int(s.store_id), round(qty), round(trip_time, 2),
                     s.distance_km, int(festival[i])])

orders = pd.DataFrame(rows, columns=["date", "store_id", "qty", "delivery_hrs",
                                     "distance_km", "festival"])
PROMISED_HRS = 3.0
raw_rows = len(orders)

# make the data messy on purpose, like real data usually is
bad_idx = rng.choice(orders.index, int(0.02 * raw_rows), replace=False)
orders.loc[bad_idx, "qty"] = np.nan
neg_idx = rng.choice(orders.index, 30, replace=False)
orders.loc[neg_idx, "qty"] = -orders.loc[neg_idx, "qty"].abs()
big_idx = rng.choice(orders.index, 15, replace=False)
orders.loc[big_idx, "delivery_hrs"] = 99
dups = orders.sample(60, random_state=1)
orders = pd.concat([orders, dups]).reset_index(drop=True)
results["raw_rows"] = len(orders)

# ---------------------------------------------------------------------------
# STEP 2: cleaning
# ---------------------------------------------------------------------------
clean = orders.copy()
clean = clean.drop_duplicates()
n_dups = len(orders) - len(clean)
n_missing = int(clean["qty"].isna().sum())
n_neg = int((clean["qty"] < 0).sum())
n_out = int((clean["delivery_hrs"] > 12).sum())

# fill missing qty with the median of the same store (not the global median)
clean["qty"] = clean.groupby("store_id")["qty"].transform(lambda s: s.fillna(s.median()))
clean.loc[clean["qty"] < 0, "qty"] = np.nan          # negative qty is wrong
clean["qty"] = clean.groupby("store_id")["qty"].transform(lambda s: s.fillna(s.median()))
# delivery time over 12 hours is a typing / system error, cap using median
med_hrs = clean.loc[clean.delivery_hrs <= 12, "delivery_hrs"].median()
clean.loc[clean.delivery_hrs > 12, "delivery_hrs"] = med_hrs
results["cleaning"] = dict(duplicates=n_dups, missing_qty=n_missing,
                           negative_qty=n_neg, absurd_delivery_time=n_out,
                           clean_rows=len(clean))

# ---------------------------------------------------------------------------
# STEP 3: KPIs
# ---------------------------------------------------------------------------
clean["on_time"] = (clean.delivery_hrs <= PROMISED_HRS).astype(int)
otd = clean.on_time.mean() * 100
avg_delivery = clean.delivery_hrs.mean()
results["kpi_on_time_delivery_pct"] = round(otd, 1)
results["kpi_avg_delivery_hrs"] = round(avg_delivery, 2)

# ---------------------------------------------------------------------------
# STEP 4: EDA charts
# ---------------------------------------------------------------------------
daily = clean.groupby("date").agg(demand=("qty", "sum"), festival=("festival", "max")).reset_index()

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(daily.date, daily.demand, lw=0.8, color="#1f4e79")
ax.plot(daily.date, daily.demand.rolling(7).mean(), lw=2, color="#e07b00", label="7-day average")
f = daily[daily.festival == 1]
ax.scatter(f.date, f.demand, s=10, color="red", label="festival days", zorder=3)
ax.set_title("Total daily demand across all stores (simulated 2025)")
ax.set_ylabel("units")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_daily_demand.png", dpi=160)
plt.close(fig)

dow = clean.assign(dow=clean.date.dt.dayofweek).groupby("dow").qty.mean()
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color="#1f4e79")
ax.set_title("Average order size by weekday")
ax.set_ylabel("units per order")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_weekday.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# STEP 5: demand forecasting (regression)
# ---------------------------------------------------------------------------
d = daily.copy()
d["dow"] = d.date.dt.dayofweek
d["month"] = d.date.dt.month
d["is_weekend"] = d.dow.isin([5, 6]).astype(int)
d["lag1"] = d.demand.shift(1)
d["lag7"] = d.demand.shift(7)
d["roll7"] = d.demand.shift(1).rolling(7).mean()
d = d.dropna().reset_index(drop=True)

features = ["dow", "month", "is_weekend", "festival", "lag1", "lag7", "roll7"]
split = len(d) - 60                       # last 60 days are the test set
train, test = d.iloc[:split], d.iloc[split:]


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


naive_pred = test["lag7"].values           # "same as last week" baseline
lr = LinearRegression().fit(train[features], train.demand)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(train[features], train.demand)
pred_lr = lr.predict(test[features])
pred_rf = rf.predict(test[features])

results["forecast"] = {
    "naive_mape": round(mape(test.demand.values, naive_pred), 2),
    "linear_mape": round(mape(test.demand.values, pred_lr), 2),
    "forest_mape": round(mape(test.demand.values, pred_rf), 2),
    "linear_mae_units": round(mean_absolute_error(test.demand, pred_lr), 1),
}

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(test.date, test.demand, label="actual", color="#1f4e79")
ax.plot(test.date, pred_lr, label="linear regression", color="#e07b00", ls="--")
ax.plot(test.date, naive_pred, label="naive (last week)", color="grey", lw=0.8)
ax.set_title("Forecast vs actual demand on the 60-day test period")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_ylabel("units")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_forecast.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# STEP 6: clustering the stores
# ---------------------------------------------------------------------------
store_stats = clean.groupby("store_id").agg(
    avg_qty=("qty", "mean"), qty_std=("qty", "std"),
    distance_km=("distance_km", "first"), avg_hrs=("delivery_hrs", "mean")).reset_index()
X = StandardScaler().fit_transform(store_stats[["avg_qty", "qty_std", "distance_km"]])
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
store_stats["cluster"] = km.labels_
order = store_stats.groupby("cluster").avg_qty.mean().sort_values().index
names = {order[0]: "Small", order[1]: "Medium", order[2]: "Large"}
store_stats["segment"] = store_stats.cluster.map(names)
results["clusters"] = store_stats.groupby("segment").agg(
    stores=("store_id", "count"), avg_qty=("avg_qty", "mean"),
    avg_distance=("distance_km", "mean")).round(1).to_dict("index")

# ---------------------------------------------------------------------------
# STEP 7: inventory policy simulation (reorder point + safety stock)
# ---------------------------------------------------------------------------
dd = daily.demand.values
mu, sigma = dd[:300].mean(), dd[:300].std()
LEAD = 3            # supplier usually delivers in 3 days, sometimes 4
LEAD_AVG, LEAD_STD = 3.5, 0.5


def simulate(rop, order_up_to, days=dd, seed=7):
    """(s, S) policy: when stock + stuff already on the way <= rop, order up to S."""
    r = np.random.default_rng(seed)
    stock, pipeline, stockout_days, stock_log = order_up_to, [], 0, []
    for demand in days:
        pipeline = [(t - 1, q) for t, q in pipeline]
        stock += sum(q for t, q in pipeline if t <= 0)
        pipeline = [(t, q) for t, q in pipeline if t > 0]
        if demand > stock:
            stockout_days += 1
            stock = 0
        else:
            stock -= demand
        position = stock + sum(q for _, q in pipeline)
        if position <= rop:
            lead = LEAD + int(r.integers(0, 2))          # supplier is sometimes 1 day late
            pipeline.append((lead, order_up_to - position))
        stock_log.append(stock)
    return stockout_days / len(days) * 100, float(np.mean(stock_log))


rop_old = mu * LEAD_AVG                                # no safety stock
z = 1.65                                               # about 95% service level
# safety stock must cover both demand swings AND supplier delay swings
safety = z * np.sqrt(LEAD_AVG * sigma ** 2 + (mu * LEAD_STD) ** 2)
rop_new = mu * LEAD_AVG + safety
so_old, inv_old = simulate(rop_old, rop_old + mu * 2)
so_new, inv_new = simulate(rop_new, rop_new + mu * 2)
results["inventory"] = {
    "mean_daily_demand": round(mu), "daily_std": round(sigma),
    "safety_stock_units": round(safety),
    "rop_old": round(rop_old), "rop_new": round(rop_new),
    "stockout_pct_old": round(so_old, 1), "stockout_pct_new": round(so_new, 1),
    "avg_stock_old": round(inv_old), "avg_stock_new": round(inv_new),
}

# ---------------------------------------------------------------------------
# STEP 8: route optimisation (capacitated vehicle routing, OR-Tools)
# ---------------------------------------------------------------------------
day_demand = (clean[clean.date == "2025-11-14"] if False else
              clean.groupby("store_id").qty.mean().round()).astype(int)   # typical day
coords = [(0, 0)] + list(zip(stores.x_km, stores.y_km))
demands = [0] + [int(day_demand[s]) for s in stores.store_id]
N_VANS, CAP = 5, 450
n = len(coords)
dist = [[int(np.hypot(a[0] - b[0], a[1] - b[1]) * ROAD_FACTOR * 100) for b in coords] for a in coords]


def route_length(route):
    return sum(dist[a][b] for a, b in zip(route[:-1], route[1:])) / 100


# baseline: how a decent dispatcher might do it by hand - sort stores by the
# direction they lie in from the warehouse (a "sweep"), then fill a van until it
# is full and start the next one. No computer optimisation here.
angle = {i: np.arctan2(coords[i][1], coords[i][0]) for i in range(1, n)}
base_routes, cur, load = [], [0], 0
for node in sorted(range(1, n), key=lambda i: angle[i]):
    if load + demands[node] > CAP:
        base_routes.append(cur + [0]); cur, load = [0], 0
    cur.append(node); load += demands[node]
base_routes.append(cur + [0])
base_km = sum(route_length(r) for r in base_routes)

mgr = pywrapcp.RoutingIndexManager(n, N_VANS, 0)
routing = pywrapcp.RoutingModel(mgr)
cb = routing.RegisterTransitCallback(lambda i, j: dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
routing.SetArcCostEvaluatorOfAllVehicles(cb)
dem_cb = routing.RegisterUnaryTransitCallback(lambda i: demands[mgr.IndexToNode(i)])
routing.AddDimensionWithVehicleCapacity(dem_cb, 0, [CAP] * N_VANS, True, "Capacity")
params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
params.time_limit.seconds = 10
sol = routing.SolveWithParameters(params)

opt_routes = []
for v in range(N_VANS):
    idx, r = routing.Start(v), [0]
    while not routing.IsEnd(idx):
        idx = sol.Value(routing.NextVar(idx))
        r.append(mgr.IndexToNode(idx))
    if len(r) > 2:
        opt_routes.append(r)
opt_km = sum(route_length(r) for r in opt_routes)

COST_PER_KM, FIXED_PER_VAN = 18, 900         # assumptions in rupees
results["routing"] = {
    "total_demand_units": int(sum(demands)),
    "vans_baseline": len(base_routes), "vans_optimised": len(opt_routes),
    "km_baseline": round(base_km, 1), "km_optimised": round(opt_km, 1),
    "km_saved_pct": round((base_km - opt_km) / base_km * 100, 1),
    "daily_cost_baseline": round(base_km * COST_PER_KM + len(base_routes) * FIXED_PER_VAN),
    "daily_cost_optimised": round(opt_km * COST_PER_KM + len(opt_routes) * FIXED_PER_VAN),
}


def draw(ax, routes, title):
    colors = ["#1f4e79", "#e07b00", "#2e8b57", "#b22222", "#6a5acd"]
    for k, r in enumerate(routes):
        xs = [coords[i][0] for i in r]; ys = [coords[i][1] for i in r]
        ax.plot(xs, ys, "-o", color=colors[k % 5], ms=4, lw=1.4)
    ax.scatter(0, 0, marker="s", s=90, color="black", zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("km east"); ax.set_ylabel("km north")


fig, axes = plt.subplots(1, 2, figsize=(9, 4))
draw(axes[0], base_routes, f"Manual sweep routes: {base_km:.0f} km")
draw(axes[1], opt_routes, f"Optimised routes: {opt_km:.0f} km")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_routes.png", dpi=160)
plt.close(fig)

with open(f"{OUT}/results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print(json.dumps(results, indent=2, default=float))
