"""Chennai demo routes shared by the dataset, API, and live map."""

ROUTES = {
    "R1": {
        "name": "Chennai Central - Avadi",
        "stops": ["Chennai Central", "Perambur", "Villivakkam", "Ambattur", "Avadi"],
        "coordinates": [(13.0827, 80.2754), (13.1167, 80.2330), (13.1080, 80.2068), (13.1142, 80.1548), (13.1155, 80.1010)],
    },
    "R2": {
        "name": "Tambaram - Chennai Beach",
        "stops": ["Tambaram", "Pallavaram", "Guindy", "Egmore", "Chennai Beach"],
        "coordinates": [(12.9249, 80.1000), (12.9675, 80.1495), (13.0067, 80.2206), (13.0787, 80.2614), (13.0966, 80.2925)],
    },
    "R3": {
        "name": "Central - Besant Nagar",
        "stops": ["Chennai Central", "Egmore", "Mylapore", "Adyar", "Besant Nagar"],
        "coordinates": [(13.0827, 80.2754), (13.0787, 80.2614), (13.0339, 80.2699), (13.0012, 80.2565), (12.9992, 80.2667)],
    },
    "R4": {
        "name": "Koyambedu - Ambattur",
        "stops": ["CMBT Koyambedu", "Anna Nagar Tower", "Thirumangalam", "Korattur", "Pattaravakkam"],
        "coordinates": [(13.0694, 80.2048), (13.0850, 80.2101), (13.0820, 80.1940), (13.1090, 80.1830), (13.1132, 80.1645)],
    },
}

WEATHER_OPTIONS = ["Clear", "Cloudy", "Humid", "Rain", "Heavy rain"]
TRAFFIC_OPTIONS = ["Light", "Moderate", "Heavy"]
