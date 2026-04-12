"""
backend/calculator.py
─────────────────────
Safe eval để tính biểu thức vật lý từ tool calling.
Có sẵn toàn bộ hằng số vật lý chương trình THPT 10-11-12.

THAY ĐỔI so với phiên bản cũ:
  - Thêm CALCULATOR_TOOL_SCHEMA   : định nghĩa tool cho Groq tool calling
  - Thêm handle_tool_call()       : entry point xử lý tool call từ LLM
  - Không thay đổi logic safe_eval (vẫn hoạt động độc lập)
"""

import math
import json

# ══════════════════════════════════════════════════════════════════
# HẰNG SỐ VẬT LÝ — THPT 10 / 11 / 12
# ══════════════════════════════════════════════════════════════════

_PHYSICS_CONSTANTS = {
    # ── LỚP 10 — Cơ học ──────────────────────────────────────────
    "g":        9.8,          # gia tốc trọng trường (m/s²) — dùng 9.8 theo SGK VN
    "g_10":     10.0,         # g = 10 m/s² (nhiều bài cho phép làm tròn)
    "G":        6.674e-11,    # hằng số hấp dẫn (N·m²/kg²)

    # ── LỚP 11 — Điện & Từ ───────────────────────────────────────
    "k":        9e9,          # hằng số Coulomb = 1/(4πε₀) (N·m²/C²)
    "k_e":      9e9,          # alias của k
    "eps0":     8.854e-12,    # hằng số điện môi chân không ε₀ (F/m)
    "mu0":      4 * math.pi * 1e-7,  # độ từ thẩm chân không μ₀ (H/m) ≈ 4π×10⁻⁷
    "e":        1.6e-19,      # điện tích nguyên tố (C)
    "m_e":      9.109e-31,    # khối lượng electron (kg)
    "m_p":      1.673e-27,    # khối lượng proton (kg)
    "m_n":      1.675e-27,    # khối lượng neutron (kg)

    # ── LỚP 12 — Dao động, Sóng, Lượng tử, Hạt nhân ─────────────
    "c":        3e8,          # tốc độ ánh sáng trong chân không (m/s)
    "h":        6.626e-34,    # hằng số Planck (J·s)
    "h_eV":     4.136e-15,    # hằng số Planck tính theo eV·s
    "eV":       1.6e-19,      # 1 eV tính ra Jun (J)
    "MeV":      1.6e-13,      # 1 MeV tính ra Jun (J)
    "u":        1.66054e-27,  # đơn vị khối lượng nguyên tử u (kg)
    "u_MeV":    931.5,        # 1u = 931.5 MeV/c²  (dùng tính năng lượng liên kết)
    "N_A":      6.022e23,     # số Avogadro (mol⁻¹)
    "k_B":      1.38e-23,     # hằng số Boltzmann (J/K)
    "R":        8.314,        # hằng số khí lý tưởng (J/mol·K)
    "sigma_sb": 5.67e-8,      # hằng số Stefan-Boltzmann (W/m²·K⁴)

    # ── Hằng số toán học ─────────────────────────────────────────
    "pi":       math.pi,
    "PI":       math.pi,
    "e_math":   math.e,       # hằng số Euler (tránh nhầm với điện tích e)
}

# ══════════════════════════════════════════════════════════════════
# HÀM TOÁN HỌC AN TOÀN
# ══════════════════════════════════════════════════════════════════

_SAFE_FUNCTIONS = {
    # Căn và lũy thừa
    "sqrt":  math.sqrt,
    "cbrt":  lambda x: x ** (1/3),
    "pow":   pow,
    "abs":   abs,
    "round": round,

    # Lượng giác (radian)
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "atan2": math.atan2,

    # Lượng giác (độ) — tiện cho bài góc 30°, 45°, 60°
    "sind":  lambda x: math.sin(math.radians(x)),
    "cosd":  lambda x: math.cos(math.radians(x)),
    "tand":  lambda x: math.tan(math.radians(x)),
    "asind": lambda x: math.degrees(math.asin(x)),
    "acosd": lambda x: math.degrees(math.acos(x)),
    "atand": lambda x: math.degrees(math.atan(x)),

    # Log và exp
    "log":   math.log,
    "log10": math.log10,
    "log2":  math.log2,
    "exp":   math.exp,

    # Chuyển đổi góc
    "deg":   math.degrees,
    "rad":   math.radians,

    # Misc
    "floor": math.floor,
    "ceil":  math.ceil,
    "min":   min,
    "max":   max,
}

_SAFE_GLOBALS = {
    "__builtins__": {},
    "math": math,
    **_PHYSICS_CONSTANTS,
    **_SAFE_FUNCTIONS,
}

_BANNED_KEYWORDS = [
    "import", "exec", "eval", "open", "print",
    "__", "os.", "sys.", "subprocess", "globals",
    "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "compile", "input", "breakpoint",
]

# ══════════════════════════════════════════════════════════════════
# CONSTANTS_HINT — nhúng vào system prompt để LLM biết tên hằng số
# ══════════════════════════════════════════════════════════════════

CONSTANTS_HINT = """Hằng số có sẵn trong calculator (dùng đúng tên):
LỚP 10 : g=9.8  g_10=10  G=6.674e-11
LỚP 11 : k=9e9  eps0=8.854e-12  mu0=4π×10⁻⁷  e=1.6e-19
         m_e=9.109e-31  m_p=1.673e-27  m_n=1.675e-27
LỚP 12 : c=3e8  h=6.626e-34  eV=1.6e-19  MeV=1.6e-13
         u=1.66054e-27  u_MeV=931.5  N_A=6.022e23
         k_B=1.38e-23  R=8.314
TOÁN   : pi  sqrt  cbrt  sin/cos/tan(rad)  sind/cosd/tand(độ)
Ví dụ  : sqrt(2*g*20)  |  k*2e-6*3e-6/0.1**2  |  h*c/500e-9/eV"""


# ══════════════════════════════════════════════════════════════════
# SAFE EVAL CORE
# ══════════════════════════════════════════════════════════════════

def safe_eval(expression: str) -> tuple[bool, str]:
    """
    Tính biểu thức toán học vật lý an toàn.

    Hỗ trợ:
      - Hằng số vật lý THPT (g, k, c, h, e, u, ...)
      - Hàm toán học (sqrt, sin, cos, sind, cosd, ...)
      - Lũy thừa dùng ** hoặc ^ (tự convert)
      - Số thập phân dùng dấu phẩy kiểu VN (tự convert)

    Returns:
        (True,  "kết quả")        nếu thành công
        (False, "thông báo lỗi") nếu thất bại
    """
    expr = expression.strip()

    # Làm sạch cú pháp
    expr = expr.replace("^", "**")
    expr = expr.replace(",", ".")
    expr = expr.replace("×", "*")
    expr = expr.replace("÷", "/")

    # Kiểm tra từ khoá nguy hiểm
    expr_lower = expr.lower()
    for banned in _BANNED_KEYWORDS:
        if banned in expr_lower:
            return False, f"Biểu thức chứa từ bị cấm: '{banned}'"

    try:
        result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307

        if isinstance(result, complex):
            return False, "Kết quả là số phức — kiểm tra lại biểu thức"

        result = float(result)

        if result != result:
            return False, "Kết quả không xác định (NaN)"

        # Format kết quả
        if result == 0.0:
            formatted = "0"
        elif abs(result) >= 1e10 or (abs(result) < 1e-4 and result != 0):
            formatted = f"{result:.4e}"
        elif result == int(result) and abs(result) < 1e9:
            formatted = str(int(result))
        else:
            formatted = f"{result:.4f}".rstrip("0").rstrip(".")

        return True, formatted

    except ZeroDivisionError:
        return False, "Lỗi: chia cho không"
    except OverflowError:
        return False, "Lỗi: kết quả quá lớn (overflow)"
    except ValueError as ve:
        return False, f"Lỗi giá trị: {ve}"
    except Exception as ex:
        return False, f"Lỗi tính toán: {ex}"


# ══════════════════════════════════════════════════════════════════
# TOOL SCHEMA — dùng cho Groq tool calling
# ══════════════════════════════════════════════════════════════════

CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Tính biểu thức toán học vật lý chính xác. "
            "Dùng khi cần tính số cụ thể trong bài tập. "
            "Hỗ trợ hằng số vật lý THPT (g=9.8, k=9e9, c=3e8, h=6.626e-34, "
            "e=1.6e-19, eV, MeV, u, u_MeV=931.5, N_A, k_B, R, ...) và "
            "hàm toán (sqrt, sin, cos, sind, cosd, tand, log, exp, ...). "
            "Ký hiệu: ** hoặc ^ cho lũy thừa, * cho nhân, / cho chia. "
            f"{CONSTANTS_HINT}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Biểu thức Python hợp lệ. "
                        "Ví dụ: 'sqrt(2*g*20)', 'k*2e-6*3e-6/0.1**2', "
                        "'h*c/500e-9/eV', '2*pi*sqrt(0.5/g)', "
                        "'u_MeV*0.03038', 'm_p + m_n'. "
                        "Không dùng dấu phẩy thập phân kiểu VN (dùng dấu chấm)."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Mô tả ngắn đang tính gì. "
                        "Ví dụ: 'vận tốc rơi tự do h=20m', "
                        "'lực Coulomb q1=2μC q2=3μC r=10cm'."
                    ),
                },
            },
            "required": ["expression", "description"],
        },
    },
}


# ══════════════════════════════════════════════════════════════════
# HANDLE TOOL CALL — entry point từ app.py
# ══════════════════════════════════════════════════════════════════

def handle_tool_call(tool_call) -> dict:
    """
    Xử lý một tool call object từ Groq response.

    Args:
        tool_call: đối tượng tool call từ response.choices[0].message.tool_calls[i]

    Returns:
        dict với keys:
          - tool_call_id : str
          - success      : bool
          - result       : str  (kết quả số hoặc thông báo lỗi)
          - expression   : str
          - description  : str
    """
    tool_call_id = tool_call.id

    try:
        args = json.loads(tool_call.function.arguments)
        expression  = args.get("expression", "").strip()
        description = args.get("description", "")

        if not expression:
            return {
                "tool_call_id": tool_call_id,
                "success": False,
                "result": "Biểu thức trống",
                "expression": expression,
                "description": description,
            }

        success, result = safe_eval(expression)
        return {
            "tool_call_id": tool_call_id,
            "success": success,
            "result": result,
            "expression": expression,
            "description": description,
        }

    except json.JSONDecodeError as e:
        return {
            "tool_call_id": tool_call_id,
            "success": False,
            "result": f"Lỗi parse arguments: {e}",
            "expression": "",
            "description": "",
        }
    except Exception as e:
        return {
            "tool_call_id": tool_call_id,
            "success": False,
            "result": f"Lỗi xử lý tool call: {e}",
            "expression": "",
            "description": "",
        }


# ══════════════════════════════════════════════════════════════════
# QUICK TEST — python backend/calculator.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # Lớp 10 — Cơ học
        ("sqrt(2*g*20)",              "v rơi tự do h=20m → ~19.8 m/s"),
        ("0.5*2*5**2",                "Eđ m=2kg v=5m/s → 25 J"),
        ("2*g_10*sind(30)",           "a mặt phẳng nghiêng 30° g=10 → 5"),
        ("cosd(60)",                  "cos60° → 0.5"),
        ("2*pi*sqrt(0.5/g)",          "T con lắc đơn l=0.5m → ~1.42s"),
        # Lớp 11 — Điện
        ("k*2e-6*3e-6/0.1**2",        "F Coulomb q1=2μC q2=3μC r=0.1m → 5.4N"),
        ("k*5e-9/0.3**2",             "E tại điểm cách q=5nC 30cm"),
        ("e*100",                     "công electron qua U=100V → 1.6e-17 J"),
        ("1/(2*pi*sqrt(1e-3*1e-6))",  "f mạch LC L=1mH C=1μF"),
        # Lớp 12 — Lượng tử & Hạt nhân
        ("h*c/500e-9/eV",             "E photon λ=500nm (eV) → ~2.48 eV"),
        ("u_MeV*0.03038",             "E liên kết từ độ hụt khối 0.03038u → ~28.3 MeV"),
        ("m_p + m_n",                 "tổng khối lượng p+n"),
        ("N_A*1.38e-23*300",          "NkT tại 300K"),
    ]

    print("=" * 65)
    print("TEST calculator.py — hằng số THPT 10/11/12")
    print("=" * 65)
    ok_count = 0
    for expr, desc in tests:
        ok, result = safe_eval(expr)
        status = "✓" if ok else "✗"
        if ok:
            ok_count += 1
        print(f"  {status} {desc}")
        print(f"      {expr} = {result}\n")
    print(f"Kết quả: {ok_count}/{len(tests)} test passed")

    # Test handle_tool_call với mock object
    print("\n" + "=" * 65)
    print("TEST handle_tool_call mock")
    print("=" * 65)

    class _MockFunction:
        arguments = json.dumps({"expression": "k*2e-6*3e-6/0.1**2", "description": "Lực Coulomb"})

    class _MockToolCall:
        id = "test_001"
        function = _MockFunction()

    r = handle_tool_call(_MockToolCall())
    print(f"  success   : {r['success']}")
    print(f"  expression: {r['expression']}")
    print(f"  result    : {r['result']}")
    print(f"  desc      : {r['description']}")