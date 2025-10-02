import {getRequest} from "/scripts/utils/requests.js";

export let objectTypes = {};

export let blendModes = {};

export let pipelineModules = {};

export let pipelineModuleIcons = {};

export let videoFilters = {};

export async function initializeData() {
    for (let name of await getRequest("/api/object/types")) {
        objectTypes[name] = await getRequest(`/api/object/${name}/schema`);
    }

    for (let name of await getRequest("/api/object/modules.blend_modes.BlendMode/subtypes")) {
        blendModes[name] = (await getRequest(`/api/object/${name}/schema`)).name;
    }

    for (let name of await getRequest("/api/object/modules.pipeline_module.PipelineModule/subtypes")) {
        pipelineModules[name] = await getRequest(`/api/object/${name}/schema`);

        for (let [start, icon] of Object.entries({
            // TODO: Add control modules' icon here
            "modules.pipeline_modules.filtering": "\u{f890}",
            "modules.pipeline_modules.measuring": "\u{f201}",
            "modules.pipeline_modules.neural": "\u{e0c6}",
            "modules.pipeline_modules.painting": "\u{f1fc}",
            "modules.pipeline_modules.temporal": "\u{f017}",
            "modules.pipeline_modules.tool": "\u{f0ad}",
            "": "\u{f013}",
        })) {
            if (name.startsWith(start)) {
                pipelineModuleIcons[name] = icon;
                break;
            }
        }
    }

    for (let name of await getRequest("/api/object/modules.video_filter.VideoFilter/subtypes")) {
        videoFilters[name] = await getRequest(`/api/object/${name}/schema`);
    }
}
