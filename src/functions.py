from .func_agent import tool

@tool
def fn_add_numbers(a: int, b: int) -> int:
    return a + b

@tool
def fn_greet(name: str):
    return f"Greetings!, {name}!"

@tool
def get_current_weather(city: str):
    if "san francisco" in city.lower():
        return {"temperature": "15°C", "condition": "Cloudy"}
    elif "new york" in city.lower():
        return {"temperature": "25°C", "condition": "Sunny"}
    else:
        return {"temperature": "20°C", "condition": "Partly Cloudy"}
