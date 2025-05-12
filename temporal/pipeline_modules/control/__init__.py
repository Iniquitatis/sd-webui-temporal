from temporal.pipeline_module import PipelineModule


# NOTE: In fact, it should be a subpipeline, as in contain a `Pipeline` instance
# of its own. Otherwise, no conveniences like sending previews or what have you
# are available here.
class ControlModule(PipelineModule, abstract = True):
    pass
