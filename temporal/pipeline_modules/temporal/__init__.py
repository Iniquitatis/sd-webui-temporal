from temporal.pipeline_module import PipelineModule


class TemporalModule(PipelineModule, abstract = True):
    is_sampleable = True
    sample_iterations = 10
