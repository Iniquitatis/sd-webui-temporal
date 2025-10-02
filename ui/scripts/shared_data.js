import {getRequest} from "../scripts/utils/requests.js";

export let objectTypes = {};

export let blendModes = {};

export let pipelineModules = {};

export let pipelineModuleIcons = {};

export let videoFilters = {};

export async function initializeData() {
    for (let name of await getRequest("/temporal/object/types")) {
        objectTypes[name] = await getRequest(`/temporal/object/${name}/schema`);
    }

    for (let name of await getRequest("/temporal/object/temporal.blend_modes.BlendMode/subtypes")) {
        blendModes[name] = (await getRequest(`/temporal/object/${name}/schema`)).name;
    }

    for (let name of await getRequest("/temporal/object/temporal.pipeline_module.PipelineModule/subtypes")) {
        pipelineModules[name] = await getRequest(`/temporal/object/${name}/schema`);

        for (let [start, icon] of Object.entries({
            // TODO: Add control modules' icon here
            "temporal.pipeline_modules.filtering": "\u{f890}",
            "temporal.pipeline_modules.measuring": "\u{f201}",
            "temporal.pipeline_modules.neural": "\u{e0c6}",
            "temporal.pipeline_modules.painting": "\u{f1fc}",
            "temporal.pipeline_modules.temporal": "\u{f017}",
            "temporal.pipeline_modules.tool": "\u{f0ad}",
            "": "\u{f013}",
        })) {
            if (name.startsWith(start)) {
                pipelineModuleIcons[name] = icon;
                break;
            }
        }
    }

    for (let name of await getRequest("/temporal/object/temporal.video_filter.VideoFilter/subtypes")) {
        videoFilters[name] = await getRequest(`/temporal/object/${name}/schema`);
    }
}
