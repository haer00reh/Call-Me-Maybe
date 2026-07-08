from func_agent import tool

@tool
def fn_add_numbers(a: float, b: float) -> float:
    return a + b


@tool
def fn_subtract_numbers(a: float, b: float) -> float:
    return a - b


@tool
def fn_multiply_numbers(a: float, b: float) -> float:
    return a * b

@tool
def fn_greet(name: str):
    return f"Hello, {name}!"


@tool
def fn_reverse_string(text: str) -> str:
    return text[::-1]

@tool
def fn_divide(a: float, b: float) -> float:
    return round(a / b, 2)


@tool
def get_current_weather(city: str):
    if "san francisco" in city.lower():
        return {"temperature": "15°C", "condition": "Cloudy"}
    elif "new york" in city.lower():
        return {"temperature": "25°C", "condition": "Sunny"}
    else:
        return {"temperature": "20°C", "condition": "Partly Cloudy"}
