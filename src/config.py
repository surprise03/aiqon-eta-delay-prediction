"""Static world definition for the shipment simulator.

All durations are in HOURS. Numbers are stylised but chosen to be in the
range seen on real container lanes (transit times, dwell, customs clearance).
"""

ROUTES = {
    # route_id: origin port, destination port, ocean transit (days), chokepoint, inland drive (h)
    "SHA-LAX": dict(origin="Shanghai", dest="Los Angeles", transit_days=15, chokepoint=None, inland_hours=30),
    "PUS-LGB": dict(origin="Busan", dest="Long Beach", transit_days=12, chokepoint=None, inland_hours=30),
    "NGB-NYC": dict(origin="Ningbo", dest="New York", transit_days=34, chokepoint="panama", inland_hours=14),
    "SZX-RTM": dict(origin="Shenzhen", dest="Rotterdam", transit_days=32, chokepoint="suez", inland_hours=12),
    "SIN-HAM": dict(origin="Singapore", dest="Hamburg", transit_days=26, chokepoint="suez", inland_hours=20),
    "NSA-FXT": dict(origin="Nhava Sheva", dest="Felixstowe", transit_days=24, chokepoint="suez", inland_hours=10),
}

# Origin ports: mean congestion (0..1)
ORIGIN_PORTS = {
    "Shanghai": 0.55, "Busan": 0.35, "Ningbo": 0.50,
    "Shenzhen": 0.50, "Singapore": 0.40, "Nhava Sheva": 0.60,
}

# Destination ports: mean congestion, planned customs hours, base inspection probability
DEST_PORTS = {
    "Los Angeles": dict(congestion=0.62, customs_hours=36, inspection_p=0.06),
    "Long Beach": dict(congestion=0.58, customs_hours=36, inspection_p=0.06),
    "New York": dict(congestion=0.50, customs_hours=30, inspection_p=0.07),
    "Rotterdam": dict(congestion=0.45, customs_hours=24, inspection_p=0.05),
    "Hamburg": dict(congestion=0.55, customs_hours=28, inspection_p=0.05),
    "Felixstowe": dict(congestion=0.52, customs_hours=26, inspection_p=0.06),
}

# Carrier schedule reliability (share of vessels on time, published metric)
CARRIERS = {"OceanA": 0.85, "BlueLine": 0.75, "TransPac": 0.65, "Meridian": 0.55}

# Cargo type -> customs inspection multiplier
CARGO = {"general": 1.0, "electronics": 1.3, "perishable": 2.5, "hazmat": 3.5}

# Chokepoint disruption: probability and added hours (low, high)
CHOKEPOINTS = {
    "suez": dict(p=0.10, hours=(200, 290)),   # Cape of Good Hope diversion ~8-12 days
    "panama": dict(p=0.07, hours=(48, 144)),  # draft restrictions / queueing
}

# Planned (scheduled) stage durations
PLAN = dict(
    berth_wait=12,        # planned anchorage -> berth
    discharge=30,         # planned berth -> container discharged
    appointment_buffer=12,  # buffer between planned customs release and truck pickup
)

LATE_THRESHOLD_H = 24  # a shipment is "delayed" if it is delivered >24h after schedule

PEAK_MONTHS = {1, 8, 9, 10}  # pre-Lunar-New-Year rush + Q3 peak season
