CONFORMAL_HALF_WIDTH = 3.152006585363337
NOMINAL_COVERAGE = .90
EMPIRICAL_COVERAGE = .9121844127332601

def prediction_interval(point: float) -> tuple[float,float]: return point-CONFORMAL_HALF_WIDTH, point+CONFORMAL_HALF_WIDTH

def interval_confidence(a: tuple[float,float], b: tuple[float,float]) -> tuple[str,float]:
    overlap=max(0.0,min(a[1],b[1])-max(a[0],b[0])); small=min(a[1]-a[0],b[1]-b[0])
    ratio=overlap/small if small>0 else 1.0
    return ("HIGH" if ratio==0 else "MODERATE" if ratio<=.5 else "LOW"),ratio
