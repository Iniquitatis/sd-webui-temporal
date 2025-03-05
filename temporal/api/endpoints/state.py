from pydantic import BaseModel

from temporal.api.endpoint import Endpoint


class _(Endpoint):
    method = "GET"
    path = "/temporal/state"

    class Response(BaseModel):
        state: str
        current_iteration: int
        total_iterations: int

    async def do(self) -> Response:
        with self.engine._state_lock:
            return self.Response(
                state = self.engine.state.state,
                current_iteration = self.engine.state.current_iteration,
                total_iterations = self.engine.state.total_iterations,
            )
