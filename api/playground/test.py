from pydantic import BaseModel


class ProductDto(BaseModel):
    name: str
    price: int


def small_function(request: ProductDto) -> str:
    return f"Hello from {request.price}"
