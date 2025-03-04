from temporal.pipeline_modules.filtering import ImageFilter


class NeuralModule(ImageFilter, abstract = True):
    is_sampleable = False
