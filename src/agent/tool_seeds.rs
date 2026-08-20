//! Bundled starter tools — scientific/teaching utility scripts seeded on first
//! run so the Tools panel isn't empty and the agent is immediately useful.
//!
//! Each entry is a `SeedTool` describing a script the `UserScriptTool` runtime
//! executes (interpreter + body + JSON schema). The contract (see
//! `agent/tools/user_script.rs`) is: the model's arguments arrive as a JSON
//! object on stdin; the script writes its result to stdout.
//!
//! Tools are grouped by scientific purpose:
//!  - **math** : a register of preloaded operations for physicists (units,
//!    constants, linear algebra, calculus, equation solving).
//!  - **viz**  : 3D modeler / plotting (generates files the user can open).
//!  - **web**  : web search + fetch (grounding for research).
//!  - **chem** : periodic table, molar mass, reaction balancing (chemistry).
//!  - **bio**  : DNA/RNA/protein sequence analysis (genetics).
//!  - **astro**: solar-system body data (astronomy).
//!  - **image**: image file analysis (dimensions, format, dominant colors).
//!  - **util** : common agent helpers (hashing).

/// A bundled starter tool.
pub struct SeedTool {
    pub name: &'static str,
    pub description: &'static str,
    /// `"python" | "node" | "sh" | "powershell"`.
    pub interpreter: &'static str,
    pub script_body: &'static str,
    /// JSON Schema (object) as a string, stored verbatim in the DB.
    pub params_schema: &'static str,
    /// `"read" | "write"`.
    pub tool_kind: &'static str,
}

/// All bundled starter tools. Seeded only when the `tools` table is empty.
/// Built as a `Vec` (not a `const`) because the script bodies are large raw
/// strings assembled by helper functions.
pub fn starter_tools() -> Vec<SeedTool> {
    vec![
        // ---- math (physicist's register) ------------------------------------
        math_constants(),
        unit_converter(),
        linear_algebra(),
        calculus(),
        equation_solver(),
        statistics(),
        // ---- viz (3D modeler / plots) ---------------------------------------
        plot_3d(),
        // ---- chem (chemistry) ------------------------------------------------
        periodic_table(),
        molar_mass(),
        // ---- bio (genetics / bioinformatics) ---------------------------------
        sequence_analysis(),
        // ---- astro (astronomy) -----------------------------------------------
        solar_system(),
        // ---- image analysis --------------------------------------------------
        image_analyzer(),
        // ---- web (research grounding) ---------------------------------------
        web_search_stub(),
        // ---- util ------------------------------------------------------------
        qr_hasher(),
    ]
}

// ---- individual tool definitions -------------------------------------------

fn math_constants() -> SeedTool {
    SeedTool {
        name: "math_constants",
        description: "Look up CODATA physical constants and common mathematical constants. Returns name, value, unit, and relative uncertainty. Useful for physics problems and teaching.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"query":{"type":"string","description":"Constant name or symbol to look up, e.g. 'speed of light', 'hbar', 'G', 'electron mass', 'pi', 'e'."}},"required":["query"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
q = data.get("query","").lower().strip()
C = {
    "speed of light": ("c", 299792458.0, "m/s", "exact"),
    "c": ("c", 299792458.0, "m/s", "exact"),
    "planck constant": ("h", 6.62607015e-34, "J*s", "exact"),
    "h": ("h", 6.62607015e-34, "J*s", "exact"),
    "reduced planck": ("hbar", 1.054571817e-34, "J*s", "exact"),
    "hbar": ("hbar", 1.054571817e-34, "J*s", "exact"),
    "gravitational constant": ("G", 6.67430e-11, "m^3 kg^-1 s^-2", "2.2e-5"),
    "g": ("G", 6.67430e-11, "m^3 kg^-1 s^-2", "2.2e-5"),
    "electron mass": ("m_e", 9.1093837015e-31, "kg", "2.9e-10"),
    "m_e": ("m_e", 9.1093837015e-31, "kg", "2.9e-10"),
    "proton mass": ("m_p", 1.67262192369e-27, "kg", "3.1e-10"),
    "m_p": ("m_p", 1.67262192369e-27, "kg", "3.1e-10"),
    "neutron mass": ("m_n", 1.67492749804e-27, "kg", "3.1e-10"),
    "m_n": ("m_n", 1.67492749804e-27, "kg", "3.1e-10"),
    "elementary charge": ("e", 1.602176634e-19, "C", "exact"),
    "e_charge": ("e", 1.602176634e-19, "C", "exact"),
    "boltzmann constant": ("k_B", 1.380649e-23, "J/K", "exact"),
    "k_b": ("k_B", 1.380649e-23, "J/K", "exact"),
    "avogadro constant": ("N_A", 6.02214076e23, "1/mol", "exact"),
    "n_a": ("N_A", 6.02214076e23, "1/mol", "exact"),
    "gas constant": ("R", 8.314462618, "J/(mol*K)", "exact"),
    "r": ("R", 8.314462618, "J/(mol*K)", "exact"),
    "permittivity of vacuum": ("eps_0", 8.8541878128e-12, "F/m", "1.5e-10"),
    "eps_0": ("eps_0", 8.8541878128e-12, "F/m", "1.5e-10"),
    "permeability of vacuum": ("mu_0", 1.25663706212e-6, "H/m", "1.7e-10"),
    "mu_0": ("mu_0", 1.25663706212e-6, "H/m", "1.7e-10"),
    "stefan-boltzmann": ("sigma", 5.670374419e-8, "W/(m^2*K^4)", "1.6e-8"),
    "sigma": ("sigma", 5.670374419e-8, "W/(m^2*K^4)", "1.6e-8"),
    "rydberg constant": ("R_inf", 10973731.568160, "1/m", "1.9e-12"),
    "bohr radius": ("a_0", 5.29177210903e-11, "m", "1.5e-10"),
    "pi": ("pi", 3.141592653589793, "-", "exact"),
    "euler": ("e", 2.718281828459045, "-", "exact"),
    "golden ratio": ("phi", 1.618033988749895, "-", "exact"),
}
if q in C:
    sym, val, unit, unc = C[q]
    print(f"{sym} = {val} {unit}  (rel. uncertainty: {unc})")
else:
    # fuzzy: find keys containing the query
    matches = [k for k in C if q in k]
    if matches:
        print("Multiple matches. Did you mean one of:")
        for m in matches:
            sym, val, unit, _ = C[m]
            print(f"  {m} -> {sym} = {val} {unit}")
    else:
        print(f"Constant '{q}' not found. Try: c, h, hbar, G, e, k_B, N_A, eps_0, sigma, a_0, pi, e...")
"#,
    }
}

fn unit_converter() -> SeedTool {
    SeedTool {
        name: "unit_converter",
        description: "Convert a numeric value between units (SI and common). Supports length, mass, time, temperature, energy, pressure, angle, velocity. Useful for physics homework and lab work.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"value":{"type":"number","description":"The numeric value to convert."},"from":{"type":"string","description":"Source unit, e.g. 'km', 'mi', 'eV', 'J', 'atm', 'degC'."},"to":{"type":"string","description":"Target unit, e.g. 'm', 'eV', 'J', 'Pa', 'degF'."}},"required":["value","from","to"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
v = float(data["value"]); frm = data["from"].strip(); to = data["to"].strip()
# Base-unit factors. Convert v -> base -> to.
L = {"m":1,"km":1e3,"cm":1e-2,"mm":1e-3,"um":1e-6,"nm":1e-9,"mi":1609.344,"yd":0.9144,"ft":0.3048,"in":0.0254,"ly":9.4607304725808e15,"au":1.495978707e11,"pc":3.0856775814913678e16}
M = {"kg":1,"g":1e-3,"mg":1e-6,"ug":1e-9,"t":1e3,"lb":0.45359237,"oz":0.028349523125,"u":1.66053906660e-27}
T = {"s":1,"ms":1e-3,"us":1e-6,"ns":1e-9,"min":60,"h":3600,"day":86400,"year":3.15576e7}
E = {"J":1,"kJ":1e3,"MJ":1e6,"cal":4.184,"kcal":4184,"eV":1.602176634e-19,"keV":1.602176634e-16,"MeV":1.602176634e-13,"Wh":3600,"kWh":3.6e6,"BTU":1055.05585262,"erg":1e-7}
P = {"Pa":1,"kPa":1e3,"MPa":1e6,"GPa":1e9,"bar":1e5,"mbar":1e2,"atm":101325,"torr":133.32236842105263,"mmHg":133.32236842105263,"psi":6894.757293168}
A = {"rad":1,"deg":0.017453292519943295,"arcmin":2.908882086657216e-4,"arcsec":4.84813681109536e-6}
V = {"m/s":1,"km/h":0.2777777777777778,"mph":0.44704,"knot":0.5144444444444444,"c":299792458.0}
groups = {"length":L,"mass":M,"time":T,"energy":E,"pressure":P,"angle":A,"velocity":V}
def find(unit):
    for name, tbl in groups.items():
        if unit in tbl: return name, tbl[unit]
    return None, None
g1,f1 = find(frm); g2,f2 = find(to)
if g1 is None: print(f"Unknown source unit '{frm}'."); sys.exit(0)
if g2 is None: print(f"Unknown target unit '{to}'."); sys.exit(0)
if g1 != g2: print(f"Cannot convert between {g1} and {g2}."); sys.exit(0)
# temperature needs offsets
if frm in ("degC","celsius") or to in ("degC","celsius") or frm in ("degF","fahrenheit") or to in ("degF","fahrenheit"):
    # normalize to Kelvin
    if frm in ("degC","celsius"): k = v + 273.15
    elif frm in ("degF","fahrenheit"): k = (v - 32)*5/9 + 273.15
    elif frm in ("K","kelvin"): k = v
    else: k = v  # assume already K
    if to in ("degC","celsius"): out = k - 273.15
    elif to in ("degF","fahrenheit"): out = (k - 273.15)*9/5 + 32
    else: out = k
    print(f"{v} {frm} = {out:g} {to}")
else:
    out = v * f1 / f2
    print(f"{v} {frm} = {out:g} {to}")
"#,
    }
}

fn linear_algebra() -> SeedTool {
    SeedTool {
        name: "linear_algebra",
        description: "Linear algebra operations on vectors and matrices: solve Ax=b, determinant, inverse, eigenvalues, matrix multiply, transpose. Great for mechanics, quantum, and teaching linear algebra.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"operation":{"type":"string","enum":["solve","det","inverse","eigenvalues","matmul","transpose"],"description":"The operation to perform."},"matrix":{"type":"array","items":{"type":"array","items":{"type":"number"}},"description":"A 2D matrix (list of rows)."},"vector":{"type":"array","items":{"type":"number"},"description":"A vector (for 'solve': the right-hand side b)."},"matrix_b":{"type":"array","items":{"type":"array","items":{"type":"number"}},"description":"Second matrix (for 'matmul')."}},"required":["operation","matrix"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
op = data["operation"]; A = data["matrix"]
try:
    import numpy as np
    A = np.array(A, dtype=float)
    if op == "det":
        print(f"det(A) = {np.linalg.det(A):.6g}")
    elif op == "inverse":
        print(np.linalg.inv(A))
    elif op == "eigenvalues":
        w, v = np.linalg.eig(A)
        for i, val in enumerate(w):
            print(f"lambda_{i+1} = {val:.6g}, eigenvector = {v[:,i].tolist()}")
    elif op == "transpose":
        print(A.T)
    elif op == "matmul":
        B = np.array(data["matrix_b"], dtype=float)
        print(A @ B)
    elif op == "solve":
        b = np.array(data["vector"], dtype=float)
        x = np.linalg.solve(A, b)
        print(f"x = {x.tolist()}")
    else:
        print(f"Unknown operation '{op}'.")
except ImportError:
    print("numpy not installed; install with: pip install numpy")
except Exception as ex:
    print(f"Error: {ex}")
"#,
    }
}

fn calculus() -> SeedTool {
    SeedTool {
        name: "calculus",
        description: "Symbolic differentiation and definite integration of a math expression string. Also numeric integration fallback. Useful for physics derivations and teaching.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"operation":{"type":"string","enum":["differentiate","integrate_definite","integrate_numeric"],"description":"The calculus operation."},"expression":{"type":"string","description":"Math expression, e.g. 'x**2 * sin(x)'. Python/sympy syntax; use ** for powers."},"variable":{"type":"string","description":"Variable to differentiate/integrate with respect to (default 'x')."},"a":{"type":"number","description":"Lower bound (definite/numeric integration)."},"b":{"type":"number","description":"Upper bound (definite/numeric integration)."}},"required":["operation","expression"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
op = data["operation"]; expr = data["expression"]; var = data.get("variable","x")
try:
    import sympy as sp
    x = sp.Symbol(var)
    e = sp.sympify(expr)
    if op == "differentiate":
        print(f"d/d{var}({expr}) = {sp.diff(e, x)}")
    elif op == "integrate_definite":
        a = float(data["a"]); b = float(data["b"])
        print(f"integral of {expr} from {a} to {b} = {sp.integrate(e, (x, a, b))}")
    elif op == "integrate_numeric":
        a = float(data["a"]); b = float(data["b"])
        from math import isfinite
        f = sp.lambdify(var, e, "math")
        n = 10000; h = (b-a)/n; s = 0.5*(f(a)+f(b))
        for i in range(1,n): s += f(a+i*h)
        print(f"numeric integral of {expr} from {a} to {b} ~= {s*h:.6f}")
    else:
        print(f"Unknown operation '{op}'.")
except ImportError:
    # numeric-only fallback for integration
    if op == "integrate_numeric":
        from math import isfinite
        a = float(data["a"]); b = float(data["b"])
        f = eval("lambda %s: %s" % (var, expr), {"__builtins__":{}}, {})
        n = 10000; h = (b-a)/n; s = 0.5*(f(a)+f(b))
        for i in range(1,n): s += f(a+i*h)
        print(f"numeric integral of {expr} from {a} to {b} ~= {s*h:.6f} (sympy not installed; no symbolic result)")
    else:
        print("sympy not installed; install with: pip install sympy (needed for symbolic ops)")
except Exception as ex:
    print(f"Error: {ex}")
"#,
    }
}

fn equation_solver() -> SeedTool {
    SeedTool {
        name: "equation_solver",
        description: "Solve an equation or system of equations symbolically. Pass one equation (e.g. 'x**2 - 4') to find roots, or an expression like 'x**2 - 4 = 0'. Great for physics problem sets.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"equation":{"type":"string","description":"An equation to solve, e.g. 'x**2 - 4' (= 0 implied) or '2*x + 3 = 7'. Python/sympy syntax."},"variable":{"type":"string","description":"Variable to solve for (default 'x')."}},"required":["equation"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
eq = data["equation"]; var = data.get("variable","x")
try:
    import sympy as sp
    x = sp.Symbol(var)
    # split on '=' if present
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        expr = sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())
    else:
        expr = sp.sympify(eq)
    sols = sp.solve(expr, x)
    if not sols:
        print(f"No solutions found for {eq}.")
    else:
        print(f"Solutions for {var} ({eq}):")
        for i, s in enumerate(sols, 1):
            print(f"  {var}_{i} = {s}")
except ImportError:
    print("sympy not installed; install with: pip install sympy")
except Exception as ex:
    print(f"Error: {ex}")
"#,
    }
}

fn statistics() -> SeedTool {
    SeedTool {
        name: "statistics",
        description: "Descriptive statistics on a list of numbers: mean, median, stddev, min, max, variance, quartiles. Pure-Python (no deps). Useful for lab data analysis and teaching.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"},"description":"List of numeric values."}},"required":["values"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json, math
data = json.load(sys.stdin)
v = [float(x) for x in data["values"]]
n = len(v)
if n == 0:
    print("No values provided."); sys.exit(0)
v_sorted = sorted(v)
mean = sum(v)/n
var = sum((x-mean)**2 for x in v)/(n-1) if n>1 else 0.0
std = math.sqrt(var)
def quantile(p):
    if n == 1: return v[0]
    k = (n-1)*p; f = int(k); c = min(f+1, n-1)
    return v_sorted[f] + (v_sorted[c]-v_sorted[f])*(k-f)
print(f"n         = {n}")
print(f"mean      = {mean:.6g}")
print(f"median    = {quantile(0.5):.6g}")
print(f"std dev   = {std:.6g}  (sample)")
print(f"variance  = {var:.6g}")
print(f"min       = {min(v):.6g}")
print(f"max       = {max(v):.6g}")
print(f"Q1 (25%)  = {quantile(0.25):.6g}")
print(f"Q3 (75%)  = {quantile(0.75):.6g}")
"#,
    }
}

fn plot_3d() -> SeedTool {
    SeedTool {
        name: "plot_3d",
        description: "3D modeler/plotter: renders a 3D surface z=f(x,y) or a parametric 3D curve and saves it as a PNG file in the working directory. Requires matplotlib+mumpy. Great for visualizing fields, potentials, and teaching multivariable calculus.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"kind":{"type":"string","enum":["surface","curve"],"description":"'surface' plots z=f(x,y); 'curve' plots a parametric 3D curve."},"expression":{"type":"string","description":"For surface: z as a function of x and y, e.g. 'sin(sqrt(x**2+y**2))'. For curve: a list of 3 expressions in t, e.g. '[cos(t), sin(t), t]'."},"filename":{"type":"string","description":"Output PNG filename (default 'plot3d.png')."},"x_range":{"type":"array","items":{"type":"number"},"description":"[min,max] for x (surface) or t (curve). Default [-5,5]."},"y_range":{"type":"array","items":{"type":"number"},"description":"[min,max] for y (surface only). Default [-5,5]."}},"required":["kind","expression"]}"#,
        tool_kind: "write",
        script_body: r#"import sys, json, os
data = json.load(sys.stdin)
kind = data["kind"]; expr = data["expression"]
fname = data.get("filename","plot3d.png")
xr = data.get("x_range",[-5,5]); yr = data.get("y_range",[-5,5])
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    if kind == "surface":
        xs = np.linspace(float(xr[0]), float(xr[1]), 60)
        ys = np.linspace(float(yr[0]), float(yr[1]), 60)
        X, Y = np.meshgrid(xs, ys)
        ctx = {"x": X, "y": Y, "sin":np.sin, "cos":np.cos, "sqrt":np.sqrt, "exp":np.exp, "pi":np.pi, "e":np.e}
        Z = eval(expr, {"__builtins__":{}}, ctx)
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"z = {expr}")
    else:
        # parametric curve: expr is a list [fx, fy, fz] in t
        parts = eval(expr) if isinstance(expr,str) else expr
        ts = np.linspace(float(xr[0]), float(xr[1]), 500)
        ctx = {"t": ts, "sin":np.sin, "cos":np.cos, "exp":np.exp, "pi":np.pi, "e":np.e}
        Xs = eval(parts[0], {"__builtins__":{}}, ctx)
        Ys = eval(parts[1], {"__builtins__":{}}, ctx)
        Zs = eval(parts[2], {"__builtins__":{}}, ctx)
        ax.plot(Xs, Ys, Zs)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"r(t) = [{parts[0]}, {parts[1]}, {parts[2]}]")
    path = os.path.abspath(fname)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved 3D plot to: {path}")
except ImportError:
    print("matplotlib+numpy not installed; install with: pip install matplotlib numpy")
except Exception as ex:
    print(f"Error: {ex}")
"#,
    }
}

fn web_search_stub() -> SeedTool {
    SeedTool {
        name: "web_search",
        description: "Search the web for current information using DuckDuckGo's HTML endpoint and return the top text results (titles + snippets + URLs). Pure-Python, no API key. Good for research grounding when the agent needs up-to-date facts.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"query":{"type":"string","description":"The search query."},"max_results":{"type":"integer","description":"Maximum number of results to return (default 5)."}},"required":["query"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json, urllib.request, urllib.parse, re
data = json.load(sys.stdin)
q = data["query"]; mx = data.get("max_results", 5)
url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Phoenix Agent)"})
try:
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
except Exception as ex:
    print(f"Search failed: {ex}"); sys.exit(0)
# DuckDuckGo HTML results: titles in <a class="result__a">, snippets in <a class="result__snippet">
titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.S)
snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
links = re.findall(r'class="result__url"[^>]*>(.*?)</a>', html, re.S)
def clean(s): return re.sub(r"<[^>]+>","",s).strip()
out = []
n = min(mx, len(titles))
if n == 0:
    print(f"No results for '{q}'.")
else:
    for i in range(n):
        t = clean(titles[i]) if i < len(titles) else ""
        s = clean(snips[i]) if i < len(snips) else ""
        l = clean(links[i]) if i < len(links) else ""
        print(f"{i+1}. {t}")
        if l: print(f"   {l}")
        if s: print(f"   {s}")
        print()
"#,
    }
}

fn qr_hasher() -> SeedTool {
    SeedTool {
        name: "text_hasher",
        description: "Compute a cryptographic hash (sha256/sha1/md5) of a string, or generate a UUID. Pure-Python (stdlib). Useful for checksums, dedup keys, and teaching about hashing.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"operation":{"type":"string","enum":["sha256","sha1","md5","uuid"],"description":"Hash algorithm, or 'uuid' to generate a random UUID."},"text":{"type":"string","description":"The input text to hash (ignored for 'uuid')."}},"required":["operation"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json, hashlib, uuid
data = json.load(sys.stdin)
op = data["operation"]; text = data.get("text","")
if op == "uuid":
    print(uuid.uuid4())
else:
    h = getattr(hashlib, op)()
    h.update(text.encode("utf-8"))
    print(f"{op}({text!r}) = {h.hexdigest()}")
"#,
    }
}

// ---- image analysis --------------------------------------------------------

fn image_analyzer() -> SeedTool {
    SeedTool {
        name: "image_analyzer",
        description: "Analyze an image file: format, dimensions (width x height), file size, color depth, mean brightness, and the top dominant colors (as hex). Uses Pillow if installed; otherwise reads basic PNG/JPEG/GIF/BMP headers from stdlib. Great for inspecting figures, diagrams, or screenshots the agent encounters.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"path":{"type":"string","description":"Path to the image file to analyze."},"dominant_colors":{"type":"integer","description":"Number of dominant colors to extract (default 5). Ignored if Pillow is not installed."}},"required":["path"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json, os
data = json.load(sys.stdin)
path = data["path"]; k = data.get("dominant_colors", 5)
if not os.path.isfile(path):
    print(f"File not found: {path}"); sys.exit(0)
size = os.path.getsize(path)
print(f"File: {path}")
print(f"Size: {size} bytes ({size/1024:.1f} KB)")
# Try Pillow for full analysis (dimensions, mode, dominant colors).
try:
    from PIL import Image
    im = Image.open(path)
    w, h = im.size
    print(f"Format: {im.format}")
    print(f"Mode  : {im.mode}")
    print(f"Dims  : {w} x {h} px")
    # Mean brightness
    gray = im.convert("L")
    px = list(gray.getdata())
    mean = sum(px)/len(px) if px else 0
    print(f"Mean brightness: {mean:.1f}/255")
    # Dominant colors via simple quantization
    small = im.convert("RGB").resize((100, 100))
    from collections import Counter
    counts = Counter(small.getdata())
    print(f"Top {k} dominant colors:")
    for (r,g,b), n in counts.most_common(k):
        print(f"  #{r:02x}{g:02x}{b:02x}  rgb({r},{g},{b})  ({n} samples)")
except ImportError:
    # Stdlib fallback: parse common header bytes for dimensions.
    with open(path, "rb") as f:
        head = f.read(32)
    fmt = "unknown"; w = h = None
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        fmt = "PNG"; w = int.from_bytes(head[16:20],"big"); h = int.from_bytes(head[20:24],"big")
    elif head[:3] == b"\xff\xd8\xff":
        fmt = "JPEG"
    elif head[:6] in (b"GIF87a", b"GIF89a"):
        fmt = "GIF"; w = int.from_bytes(head[6:8],"little"); h = int.from_bytes(head[8:10],"little")
    elif head[:2] == b"BM":
        fmt = "BMP"; w = int.from_bytes(head[18:22],"little"); h = int.from_bytes(head[22:26],"little")
    print(f"Format: {fmt}")
    if w and h:
        print(f"Dims  : {w} x {h} px")
    else:
        print("Dims  : (not parsed without Pillow)")
    print("(Install Pillow for color analysis: pip install Pillow)")
"#,
    }
}

// Allow JSON literal usage if needed elsewhere.
#[allow(dead_code)]
fn _unused() {}

// ---- chemistry -------------------------------------------------------------

fn periodic_table() -> SeedTool {
    SeedTool {
        name: "periodic_table",
        description: "Look up an element by symbol or name (e.g. 'Fe', 'iron', 'U'). Returns atomic number, atomic mass, category, and electron configuration. Pure-Python (stdlib). Useful for chemistry homework and teaching.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"query":{"type":"string","description":"Element symbol (e.g. 'Fe', 'O'), name (e.g. 'iron'), or atomic number as a string (e.g. '26')."}},"required":["query"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
q = data.get("query","").strip()
# (symbol, name, number, mass, category, config). Masses are standard atomic weights.
E = {
"H":("H","hydrogen",1,1.008,"nonmetal","1s1"),"He":("He","helium",2,4.0026,"noble gas","1s2"),
"Li":("Li","lithium",3,6.94,"alkali metal","[He] 2s1"),"Be":("Be","beryllium",4,9.0122,"alkaline earth","[He] 2s2"),
"B":("B","boron",5,10.81,"metalloid","[He] 2s2 2p1"),"C":("C","carbon",6,12.011,"nonmetal","[He] 2s2 2p2"),
"N":("N","nitrogen",7,14.007,"nonmetal","[He] 2s2 2p3"),"O":("O","oxygen",8,15.999,"nonmetal","[He] 2s2 2p4"),
"F":("F","fluorine",9,18.998,"halogen","[He] 2s2 2p5"),"Ne":("Ne","neon",10,20.180,"noble gas","[He] 2s2 2p6"),
"Na":("Na","sodium",11,22.990,"alkali metal","[Ne] 3s1"),"Mg":("Mg","magnesium",12,24.305,"alkaline earth","[Ne] 3s2"),
"Al":("Al","aluminum",13,26.982,"post-transition metal","[Ne] 3s2 3p1"),"Si":("Si","silicon",14,28.085,"metalloid","[Ne] 3s2 3p2"),
"P":("P","phosphorus",15,30.974,"nonmetal","[Ne] 3s2 3p3"),"S":("S","sulfur",16,32.06,"nonmetal","[Ne] 3s2 3p4"),
"Cl":("Cl","chlorine",17,35.45,"halogen","[Ne] 3s2 3p5"),"Ar":("Ar","argon",18,39.948,"noble gas","[Ne] 3s2 3p6"),
"K":("K","potassium",19,39.098,"alkali metal","[Ar] 4s1"),"Ca":("Ca","calcium",20,40.078,"alkaline earth","[Ar] 4s2"),
"Fe":("Fe","iron",26,55.845,"transition metal","[Ar] 3d6 4s2"),"Cu":("Cu","copper",29,63.546,"transition metal","[Ar] 3d10 4s1"),
"Zn":("Zn","zinc",30,65.38,"transition metal","[Ar] 3d10 4s2"),"Ag":("Ag","silver",47,107.87,"transition metal","[Kr] 4d10 5s1"),
"Sn":("Sn","tin",50,118.71,"post-transition metal","[Kr] 4d10 5s2 5p2"),"I":("I","iodine",53,126.90,"halogen","[Kr] 4d10 5s2 5p5"),
"Au":("Au","gold",79,196.97,"transition metal","[Xe] 4f14 5d10 6s1"),"Hg":("Hg","mercury",80,200.59,"transition metal","[Xe] 4f14 5d10 6s2"),
"Pb":("Pb","lead",82,207.2,"post-transition metal","[Xe] 4f14 5d10 6s2 6p2"),"U":("U","uranium",92,238.03,"actinide","[Rn] 5f3 6d1 7s2"),
}
# build name->symbol and number->symbol indices
by_name = {v[1]: s for s, v in E.items()}
by_num = {v[2]: s for s, v in E.items()}
sym = None
if q in E: sym = q
elif q.lower() in by_name: sym = by_name[q.lower()]
elif q.isdigit() and int(q) in by_num: sym = by_num[int(q)]
if sym is None:
    # fuzzy
    matches = [s for s, v in E.items() if q.lower() in v[1] or q.lower() in s.lower()]
    if matches:
        print(f"No exact match. Did you mean: {', '.join(matches)}?")
    else:
        print(f"Element '{q}' not found. Try a symbol like 'Fe' or name like 'iron'.")
    sys.exit(0)
s, name, num, mass, cat, cfg = E[sym]
print(f"{s} ({name})")
print(f"  atomic number : {num}")
print(f"  atomic mass   : {mass} u")
print(f"  category      : {cat}")
print(f"  configuration : {cfg}")
"#,
    }
}

fn molar_mass() -> SeedTool {
    SeedTool {
        name: "molar_mass",
        description: "Compute the molar mass of a chemical formula (e.g. 'H2O', 'C6H12O6', 'Fe2(SO4)3', 'Ca(OH)2'). Returns g/mol and a per-element breakdown. Pure-Python (stdlib). Useful for stoichiometry.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"formula":{"type":"string","description":"A chemical formula, e.g. 'H2O', 'C6H12O6', 'Fe2(SO4)3'."}},"required":["formula"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json, re
data = json.load(sys.stdin)
formula = data["formula"].strip()
# Atomic masses (subset; extend as needed). Keys are element symbols.
AM = {"H":1.008,"He":4.0026,"Li":6.94,"Be":9.0122,"B":10.81,"C":12.011,"N":14.007,"O":15.999,"F":18.998,"Ne":20.180,
"Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.085,"P":30.974,"S":32.06,"Cl":35.45,"Ar":39.948,"K":39.098,"Ca":40.078,
"Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,"Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.38,
"Ga":69.723,"Ge":72.630,"As":74.922,"Se":78.971,"Br":79.904,"Kr":83.798,"Rb":85.468,"Sr":87.62,"Y":88.906,"Zr":91.224,
"Nb":92.906,"Mo":95.95,"Tc":98,"Ru":101.07,"Rh":102.91,"Pd":106.42,"Ag":107.87,"Cd":112.41,"In":114.82,"Sn":118.71,
"Sb":121.76,"Te":127.60,"I":126.90,"Xe":131.29,"Cs":132.91,"Ba":137.33,"La":138.91,"W":183.84,"Pt":195.08,"Au":196.97,
"Hg":200.59,"Pb":207.2,"Bi":208.98,"U":238.03}
def parse(s):
    # returns list of (element, count) handling parentheses recursively
    i = 0
    tokens = []
    while i < len(s):
        c = s[i]
        if c == "(":
            # find matching close
            depth = 1; j = i+1
            while j < len(s) and depth>0:
                if s[j]=="(": depth+=1
                elif s[j]==")": depth-=1
                j += 1
            inner = s[i+1:j-1]
            # multiplier after )
            m = re.match(r"\d+", s[j:])
            mult = int(m.group()) if m else 1
            for el, cnt in parse(inner):
                tokens.append((el, cnt*mult))
            i = j + (len(m.group()) if m else 0)
        elif c.isupper():
            el = c
            i += 1
            while i < len(s) and s[i].islower():
                el += s[i]; i += 1
            m = re.match(r"\d+", s[i:])
            cnt = int(m.group()) if m else 1
            tokens.append((el, cnt))
            i += (len(m.group()) if m else 0)
        else:
            i += 1
    return tokens
try:
    counts = {}
    for el, cnt in parse(formula):
        if el not in AM:
            print(f"Unknown element '{el}'."); sys.exit(0)
        counts[el] = counts.get(el,0) + cnt
    total = sum(counts[el]*AM[el] for el in counts)
    print(f"Molar mass of {formula} = {total:.4f} g/mol")
    print("Breakdown:")
    for el in sorted(counts):
        print(f"  {el}: {counts[el]} x {AM[el]} = {counts[el]*AM[el]:.4f}")
except Exception as ex:
    print(f"Error parsing formula: {ex}")
"#,
    }
}

// ---- biology / genetics ----------------------------------------------------

fn sequence_analysis() -> SeedTool {
    SeedTool {
        name: "sequence_analysis",
        description: "Analyze a DNA, RNA, or protein sequence: nucleotide/amino-acid counts, GC content, length, molecular weight (est.), and transcription/translation. Pure-Python (stdlib). Useful for genetics and bioinformatics teaching.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"sequence":{"type":"string","description":"A biological sequence (DNA/RNA/protein) using standard one-letter codes, e.g. 'ATGCGTACCTAG'."},"type":{"type":"string","enum":["dna","rna","protein"],"description":"Sequence type (default 'dna')."}},"required":["sequence"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
seq = data["sequence"].upper().replace(" ","").replace("\n","")
typ = data.get("type","dna").lower()
n = len(seq)
if n == 0:
    print("Empty sequence."); sys.exit(0)
print(f"Sequence: {seq}")
print(f"Type    : {typ}")
print(f"Length  : {n} residues")
if typ in ("dna","rna"):
    # nucleotide composition + GC content
    from collections import Counter
    c = Counter(seq)
    print("Composition:")
    for base in "ACGTU":
        if c[base]: print(f"  {base}: {c[base]}")
    gc = c.get("G",0)+c.get("C",0)
    at = c.get("A",0)+c.get("T",0)+c.get("U",0)
    if gc+at > 0:
        print(f"GC content: {100*gc/(gc+at):.2f}%")
    # transcription (DNA -> mRNA)
    if typ == "dna":
        mrna = seq.replace("T","U")
        print(f"Transcript (mRNA): {mrna}")
        # translate to protein (standard genetic code, stops as '*')
        CODON = {
        "UUU":"F","UUC":"F","UUA":"L","UUG":"L","CUU":"L","CUC":"L","CUA":"L","CUG":"L",
        "AUU":"I","AUC":"I","AUA":"I","AUG":"M","GUU":"V","GUC":"V","GUA":"V","GUG":"V",
        "UCU":"S","UCC":"S","UCA":"S","UCG":"S","CCU":"P","CCC":"P","CCA":"P","CCG":"P",
        "ACU":"T","ACC":"T","ACA":"T","ACG":"T","GCU":"A","GCC":"A","GCA":"A","GCG":"A",
        "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*","UGA":"*","CAU":"H","CAC":"H","CAA":"Q","CAG":"Q",
        "AAU":"N","AAC":"N","AAA":"K","AAG":"K","GAU":"D","GAC":"D","GAA":"E","GAG":"E",
        "UGU":"C","UGC":"C","UGG":"W","CGU":"R","CGC":"R","CGA":"R","CGG":"R","AGU":"S","AGC":"S",
        "AGA":"R","AGG":"R","GGU":"G","GGC":"G","GGA":"G","GGG":"G"}
        prot = "".join(CODON.get(mrna[i:i+3],"?") for i in range(0, len(mrna)-len(mrna)%3, 3))
        print(f"Translated protein: {prot}")
else:
    # protein: amino acid composition
    from collections import Counter
    c = Counter(seq)
    print("Amino acid composition:")
    for aa, cnt in sorted(c.items()):
        print(f"  {aa}: {cnt}")
# approximate molecular weight
MW_NUCLEOTIDE = 330.0  # Da avg for ssDNA/RNA nucleotide
MW_AA = 110.0          # Da avg per amino acid residue
if typ in ("dna","rna"):
    print(f"Approx. molecular weight: {n*MW_NUCLEOTIDE:.0f} Da ({n*MW_NUCLEOTIDE/1000:.1f} kDa)")
else:
    print(f"Approx. molecular weight: {n*MW_AA:.0f} Da ({n*MW_AA/1000:.1f} kDa)")
"#,
    }
}

// ---- astronomy -------------------------------------------------------------

fn solar_system() -> SeedTool {
    SeedTool {
        name: "solar_system_data",
        description: "Look up a solar-system body (planet, dwarf planet, moon, or the Sun). Returns mass, radius, semi-major axis, orbital period, and rotation period. Pure-Python (stdlib). Useful for astronomy and orbital mechanics.",
        interpreter: "python",
        params_schema: r#"{"type":"object","properties":{"body":{"type":"string","description":"Body name, e.g. 'sun', 'earth', 'jupiter', 'mars', 'moon', 'ceres'."}},"required":["body"]}"#,
        tool_kind: "read",
        script_body: r#"import sys, json
data = json.load(sys.stdin)
q = data.get("body","").lower().strip()
# (mass [kg], radius [m], semi-major axis [AU], orbital period [years], rotation period [hours])
# "—" where not applicable.
B = {
"sun":("Sun",1.989e30,6.9634e8,"—","—",609.12),
"mercury":("Mercury",3.3011e23,2.4397e6,0.387,0.2408,1407.6),
"venus":("Venus",4.8675e24,6.0518e6,0.723,0.6152,-5832.5),
"earth":("Earth",5.972e24,6.371e6,1.0,1.0,23.9345),
"moon":("Moon",7.342e22,1.7371e6,0.00257,0.0748,655.7),
"mars":("Mars",6.4171e23,3.3895e6,1.524,1.881,24.6229),
"jupiter":("Jupiter",1.898e27,6.9911e7,5.203,11.862,9.9259),
"saturn":("Saturn",5.683e26,5.8232e7,9.537,29.457,10.656),
"uranus":("Uranus",8.681e25,2.5362e7,19.191,84.011,-17.24),
"neptune":("Neptune",1.024e26,2.4622e7,30.07,164.79,16.11),
"ceres":("Ceres",9.3835e20,4.73e5,2.766,4.601,9.07),
"pluto":("Pluto",1.303e22,1.1883e6,39.482,248.0,-153.3),
}
def fmt(x):
    if isinstance(x,(int,float)): return f"{x:.4g}"
    return str(x)
if q in B:
    name,m,r,a,P,rot = B[q]
    print(f"{name}")
    print(f"  mass            : {fmt(m)} kg")
    print(f"  radius          : {fmt(r)} m")
    print(f"  semi-major axis : {fmt(a)} AU")
    print(f"  orbital period  : {fmt(P)} years")
    print(f"  rotation period : {fmt(rot)} hours")
else:
    matches = [k for k in B if q in k]
    if matches:
        print(f"No exact match. Did you mean: {', '.join(matches)}?")
    else:
        print(f"Body '{q}' not found. Try: sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune, ceres, pluto.")
"#,
    }
}