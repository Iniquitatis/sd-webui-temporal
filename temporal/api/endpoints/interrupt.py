from temporal.api.endpoint import Endpoint


class _(Endpoint):
    method = "POST"
    path = "/temporal/interrupt"

    async def do(self) -> None:
        self.engine.stop()
