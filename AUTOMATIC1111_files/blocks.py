from __future__ import annotations

import copy
import inspect
import json
import os
import random
import secrets
import sys
import time
import warnings
import webbrowser
from abc import abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Set, Tuple, Type

import anyio
import requests
from anyio import CapacityLimiter
from typing_extensions import Literal

from gradio import components, external, networking, queueing, routes, strings, utils
from gradio.context import Context
from gradio.deprecation import check_deprecated_parameters
from gradio.documentation import document, set_documentation_group
from gradio.exceptions import DuplicateBlockError, InvalidApiName
from gradio.helpers import EventData, create_tracker, skip, special_args
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import CURRENT_TUNNELS
from gradio.utils import (
    GRADIO_VERSION,
    TupleNoPrint,
    check_function_inputs_match,
    component_or_layout_class,
    delete_none,
    get_cancel_function,
    get_continuous_fn,
)

set_documentation_group("blocks")

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    import comet_ml
    from fastapi.applications import FastAPI

    from gradio.components import Component


class Block:
    def __init__(
        self,
        *,
        render: bool = True,
        elem_id: str | None = None,
        elem_classes: List[str] | str | None = None,
        visible: bool = True,
        root_url: str | None = None,  # URL that is prepended to all file paths
        _skip_init_processing: bool = False,  # Used for loading from Spaces
        **kwargs,
    ):
        self._id = Context.id
        Context.id += 1
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = (
            [elem_classes] if isinstance(elem_classes, str) else elem_classes
        )
        self.root_url = root_url
        self.share_token = secrets.token_urlsafe(32)
        self._skip_init_processing = _skip_init_processing
        self._style = {}
        self.parent: BlockContext | None = None
        self.root = ""

        if render:
            self.render()
        check_deprecated_parameters(self.__class__.__name__, **kwargs)

    def render(self):
        """
        将自身添加到适当的BlockContext中。
        """
        if Context.root_block is not None and self._id in Context.root_block.blocks:
            raise DuplicateBlockError(
                f"具有id {self._id} 的块已在当前Blocks中呈现。"
            )
        if Context.block is not None:
            Context.block.add(self)
        if Context.root_block is not None:
            Context.root_block.blocks[self._id] = self
            if isinstance(self, components.TempFileManager):
                Context.root_block.temp_file_sets.append(self.temp_files)
        return self

    def unrender(self):
        """
        如果已呈现，则从BlockContext中删除自身（否则不执行任何操作）。
    从布局和块的集合中删除自身，但不删除任何事件触发器。
        """
        if Context.block is not None:
            try:
                Context.block.children.remove(self)
            except ValueError:
                pass
        if Context.root_block is not None:
            try:
                del Context.root_block.blocks[self._id]
            except KeyError:
                pass
        return self

    def get_block_name(self) -> str:
        """
        Gets block's class name.

        If it is template component it gets the parent's class name.

        @return: class name
        """
        return (
            self.__class__.__base__.__name__.lower()
            if hasattr(self, "is_template")
            else self.__class__.__name__.lower()
        )

    def get_expected_parent(self) -> Type[BlockContext] | None:
        return None

    def set_event_trigger(
        self,
        event_name: str,
        fn: Callable | None,
        inputs: Component | List[Component] | Set[Component] | None,
        outputs: Component | List[Component] | None,
        preprocess: bool = True,
        postprocess: bool = True,
        scroll_to_output: bool = False,
        show_progress: bool = True,
        api_name: str | None = None,
        js: str | None = None,
        no_target: bool = False,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        cancels: List[int] | None = None,
        every: float | None = None,
        collects_event_data: bool | None = None,
        trigger_after: int | None = None,
        trigger_only_on_success: bool = False,
    ) -> Tuple[Dict[str, Any], int]:
        """
        添加事件到组件的依赖项中。
    参数:
        event_name: 事件名称
        fn: 可调用函数
        inputs: 输入列表
        outputs: 输出列表
        preprocess: 是否运行组件的预处理方法
        postprocess: 是否运行组件的后处理方法
        scroll_to_output: 是否在触发时滚动到依赖项的输出
        show_progress: 运行时是否显示进度动画。
        api_name: 定义此参数将在api文档中公开端点
        js: 可选前端js方法，在运行'fn'之前运行。js方法的输入参数是'inputs'和'outputs'的值，返回值应是输出组件的值列表
        no_target: 如果为True，则将"targets"设置为[]，用于Blocks "load"事件
        batch: 是否该函数接受批量输入
        max_batch_size: 发送给函数的最大批处理大小
        cancels: 另一个事件列表，用于当触发此事件时取消其他事件。例如，设置cancels = [click_event]将取消click_event，其中click_event是另一个组件的返回值。.click方法。
        every: 在客户端连接打开的同时运行此事件的“每个”秒数。以秒为单位解释。必须启用队列。
        collects_event_data: 是否收集此事件的事件数据
        trigger_after: 如果设置了，此事件将在'trigger_after'函数索引之后触发。
        trigger_only_on_success: 如果为True，则仅在先前事件成功完成后触发此事件（仅适用于`trigger_after`已设置的情况）
    返回: 依赖信息，依赖索引
        """
        # 支持单数参数
        if isinstance(inputs, set):
            inputs_as_dict = True
            inputs = sorted(inputs, key=lambda x: x._id)
        else:
            inputs_as_dict = False
            if inputs is None:
                inputs = []
            elif not isinstance(inputs, list):
                inputs = [inputs]

        if isinstance(outputs, set):
            outputs = sorted(outputs, key=lambda x: x._id)
        else:
            if outputs is None:
                outputs = []
            elif not isinstance(outputs, list):
                outputs = [outputs]

        if fn is not None and not cancels:
            check_function_inputs_match(fn, inputs, inputs_as_dict)

        if Context.root_block is None:
            raise AttributeError(
                f"event_name}()和其他事件只能在Blocks上下文中调用。"
            )
        if every is not None and every <= 0:
            raise ValueError("每个参数必须是正数或空值")
        if every and batch:
            raise ValueError(
                f"无法在批处理模式下运行{event_name}事件，且每 {every} 秒运行。 "
                "batch应为True，而every应为非零但不是两者都。"
            )

        if every and fn:
            fn = get_continuous_fn(fn, every)
        elif every:
            raise ValueError("`fn`为空时不能设置`every`的值。")

        _, progress_index, event_data_index = (
            special_args(fn) if fn else (None, None, None)
        )
        Context.root_block.fns.append(
            BlockFunction(
                fn,
                inputs,
                outputs,
                preprocess,
                postprocess,
                inputs_as_dict,
                progress_index is not None,
            )
        )
        if api_name is not None:
            api_name_ = utils.append_unique_suffix(
                api_name, [dep["api_name"] for dep in Context.root_block.dependencies]
            )
            if not (api_name == api_name_):
                warnings.warn(
                    "api_name {}已经存在，使用{}".format(api_name, api_name_)
                )
                api_name = api_name_

        if collects_event_data is None:
            collects_event_data = event_data_index is not None

        dependency = {
            "targets": [self._id] if not no_target else [],
            "trigger": event_name,
            "inputs": [block._id for block in inputs],
            "outputs": [block._id for block in outputs],
            "backend_fn": fn is not None,
            "js": js,
            "queue": False if fn is None else queue,
            "api_name": api_name,
            "scroll_to_output": scroll_to_output,
            "show_progress": show_progress,
            "every": every,
            "batch": batch,
            "max_batch_size": max_batch_size,
            "cancels": cancels or [],
            "types": {
                "continuous": bool(every),
                "generator": inspect.isgeneratorfunction(fn) or bool(every),
            },
            "collects_event_data": collects_event_data,
            "trigger_after": trigger_after,
            "trigger_only_on_success": trigger_only_on_success,
        }
        Context.root_block.dependencies.append(dependency)
        return dependency, len(Context.root_block.dependencies) - 1

    def get_config(self):
        return {
            "visible": self.visible,
            "elem_id": self.elem_id,
            "elem_classes": self.elem_classes,
            "style": self._style,
            "root_url": self.root_url,
        }

    @staticmethod
    @abstractmethod
    def update(**kwargs) -> Dict:
        return {}

    @classmethod
    def get_specific_update(cls, generic_update: Dict[str, Any]) -> Dict:
        generic_update = generic_update.copy()
        del generic_update["__type__"]
        specific_update = cls.update(**generic_update)
        return specific_update


class BlockContext(Block):
    def __init__(
        self,
        visible: bool = True,
        render: bool = True,
        **kwargs,
    ):
        """
        Parameters:
            visible: If False, this will be hidden but included in the Blocks config file (its visibility can later be updated).
            render: If False, this will not be included in the Blocks config file at all.
        """
        self.children: List[Block] = []
        Block.__init__(self, visible=visible, render=render, **kwargs)

    def __enter__(self):
        self.parent = Context.block
        Context.block = self
        return self

    def add(self, child: Block):
        child.parent = self
        self.children.append(child)

    def fill_expected_parents(self):
        children = []
        pseudo_parent = None
        for child in self.children:
            expected_parent = child.get_expected_parent()
            if not expected_parent or isinstance(self, expected_parent):
                pseudo_parent = None
                children.append(child)
            else:
                if pseudo_parent is not None and isinstance(
                    pseudo_parent, expected_parent
                ):
                    pseudo_parent.children.append(child)
                else:
                    pseudo_parent = expected_parent(render=False)
                    children.append(pseudo_parent)
                    pseudo_parent.children = [child]
                    if Context.root_block:
                        Context.root_block.blocks[pseudo_parent._id] = pseudo_parent
                child.parent = pseudo_parent
        self.children = children

    def __exit__(self, *args):
        if getattr(self, "allow_expected_parents", True):
            self.fill_expected_parents()
        Context.block = self.parent

    def postprocess(self, y):
        """
        Any postprocessing needed to be performed on a block context.
        """
        return y


class BlockFunction:
    def __init__(
        self,
        fn: Callable | None,
        inputs: List[Component],
        outputs: List[Component],
        preprocess: bool,
        postprocess: bool,
        inputs_as_dict: bool,
        tracks_progress: bool = False,
    ):
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.tracks_progress = tracks_progress
        self.total_runtime = 0
        self.total_runs = 0
        self.inputs_as_dict = inputs_as_dict
        self.name = getattr(fn, "__name__", "fn") if fn is not None else None

    def __str__(self):
        return str(
            {
                "fn": self.name,
                "preprocess": self.preprocess,
                "postprocess": self.postprocess,
            }
        )

    def __repr__(self):
        return str(self)


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def postprocess_update_dict(block: Block, update_dict: Dict, postprocess: bool = True):
    """
    Converts a dictionary of updates into a format that can be sent to the frontend.
    E.g. {"__type__": "generic_update", "value": "2", "interactive": False}
    Into -> {"__type__": "update", "value": 2.0, "mode": "static"}

    Parameters:
        block: The Block that is being updated with this update dictionary.
        update_dict: The original update dictionary
        postprocess: Whether to postprocess the "value" key of the update dictionary.
    """
    if update_dict.get("__type__", "") == "generic_update":
        update_dict = block.get_specific_update(update_dict)
    if update_dict.get("value") is components._Keywords.NO_VALUE:
        update_dict.pop("value")
    interactive = update_dict.pop("interactive", None)
    if interactive is not None:
        update_dict["mode"] = "dynamic" if interactive else "static"
    prediction_value = delete_none(update_dict, skip_value=True)
    if "value" in prediction_value and postprocess:
        assert isinstance(
            block, components.IOComponent
        ), f"Component {block.__class__} does not support value"
        prediction_value["value"] = block.postprocess(prediction_value["value"])
    return prediction_value


def convert_component_dict_to_list(
    outputs_ids: List[int], predictions: Dict
) -> List | Dict:
    """
    Converts a dictionary of component updates into a list of updates in the order of
    the outputs_ids and including every output component. Leaves other types of dictionaries unchanged.
    E.g. {"textbox": "hello", "number": {"__type__": "generic_update", "value": "2"}}
    Into -> ["hello", {"__type__": "generic_update"}, {"__type__": "generic_update", "value": "2"}]
    """
    keys_are_blocks = [isinstance(key, Block) for key in predictions.keys()]
    if all(keys_are_blocks):
        reordered_predictions = [skip() for _ in outputs_ids]
        for component, value in predictions.items():
            if component._id not in outputs_ids:
                raise ValueError(
                    f"Returned component {component} not specified as output of function."
                )
            output_index = outputs_ids.index(component._id)
            reordered_predictions[output_index] = value
        predictions = utils.resolve_singleton(reordered_predictions)
    elif any(keys_are_blocks):
        raise ValueError(
            "Returned dictionary included some keys as Components. Either all keys must be Components to assign Component values, or return a List of values to assign output values in order."
        )
    return predictions


@document("launch", "queue", "integrate", "load")
class Blocks(BlockContext):
    """
    Blocks是Gradio的低级API，允许您创建比Interfaces更为自定义的Web应用程序和演示文稿（但仍完全使用Python）。与Interface类相比，Blocks提供了更多的灵活性和控制：
(1)组件布局 (2)触发函数执行的事件 (3)数据流（例如，输入可以触发输出，这些输出可以触发下一级输出）。
Blocks还提供了将相关演示文稿分组在一起（例如使用标签）的方法。
Blocks的基本用法如下：创建一个Blocks对象，然后将其用作上下文(with语句)，然后在Blocks上下文中定义布局、组件或事件。最后，调用launch()方法启动演示文稿。
以下是一个示例：

    import gradio as gr
    def update(name):
        return f"Welcome to Gradio, {name}!"
    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        with gr.Row():
            inp = gr.Textbox(placeholder="What is your name?")
            out = gr.Textbox()
        btn = gr.Button("Run")
        btn.click(fn=update, inputs=inp, outputs=out)
    demo.launch()
演示文稿: blocks_hello, blocks_flipper, blocks_speech_text_sentiment, generate_english_german, sound_alert
指南: blocks_and_event_listeners, controlling_layout, state_in_blocks, custom_CSS_and_JS, custom_interpretations_with_blocks, using_blocks_like_functions
    """

    def __init__(
        self,
        theme: Theme | str | None = None,
        analytics_enabled: bool | None = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: str | None = None,
        **kwargs,
    ):
        """
        参数：
analytics_enabled: 是否允许基本遥测。如果为None，则使用GRADIO_ANALYTICS_ENABLED环境变量或默认为True。
mode：用于创建Blocks或Interface的人类友好名称。
title：在浏览器窗口中打开时要显示的选项卡标题。
css：自定义css或应用于整个Blocks的自定义css文件的路径。
        """
        # 清理Interface的共享参数#TODO：在使用具有Blocks的Interface之后，此部分是否仍然必要？
        self.limiter = None
        self.save_to = None
        if theme is None:
            theme = DefaultTheme()
        elif isinstance(theme, str):
            try:
                theme = Theme.from_hub(theme)
            except Exception as e:
                warnings.warn(f"Cannot load {theme}. Caught Exception: {str(e)}")
                theme = DefaultTheme()
        if not isinstance(theme, Theme):
            warnings.warn("Theme should be a class loaded from gradio.themes")
            theme = DefaultTheme()
        self.theme = theme
        self.theme_css = theme._get_theme_css()
        self.stylesheets = theme._stylesheets
        self.encrypt = False
        self.share = False
        self.enable_queue = None
        self.max_threads = 40
        self.show_error = True
        if css is not None and os.path.exists(css):
            with open(css) as css_file:
                self.css = css_file.read()
        else:
            self.css = css

        # For analytics_enabled and allow_flagging: (1) first check for
        # parameter, (2) check for env variable, (3) default to True/"manual"
        self.analytics_enabled = (
            analytics_enabled
            if analytics_enabled is not None
            else os.getenv("GRADIO_ANALYTICS_ENABLED", "True") == "True"
        )
        if not self.analytics_enabled:
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "True"
        super().__init__(render=False, **kwargs)
        self.blocks: Dict[int, Block] = {}
        self.fns: List[BlockFunction] = []
        self.dependencies = []
        self.mode = mode

        self.is_running = False
        self.local_url = None
        self.share_url = None
        self.width = None
        self.height = None
        self.api_open = True

        self.is_space = True if os.getenv("SYSTEM") == "spaces" else False
        self.favicon_path = None
        self.auth = None
        self.dev_mode = True
        self.app_id = random.getrandbits(64)
        self.temp_file_sets = []
        self.title = title
        self.show_api = True

        # Only used when an Interface is loaded from a config
        self.predict = None
        self.input_components = None
        self.output_components = None
        self.__name__ = None
        self.api_mode = None
        self.progress_tracking = None

        self.file_directories = []

        if self.analytics_enabled:
            data = {
                "mode": self.mode,
                "custom_css": self.css is not None,
                "theme": self.theme,
                "version": GRADIO_VERSION,
            }
            utils.initiated_analytics(data)

    @classmethod
    def from_config(
        cls,
        config: dict,
        fns: List[Callable],
        root_url: str | None = None,
    ) -> Blocks:
        """
        工厂方法，用于从配置和函数列表创建Blocks。

    参数：
    config：包含Blocks配置的字典。
    fns：在Blocks中使用的函数列表。必须按照配置中的依赖项顺序排列。
    root_url：可选的根URL，用于Blocks中的组件。允许从外部URL提供文件。
        """
        config = copy.deepcopy(config)
        components_config = config["components"]
        original_mapping: Dict[int, Block] = {}

        def get_block_instance(id: int) -> Block:
            for block_config in components_config:
                if block_config["id"] == id:
                    break
            else:
                raise ValueError("无法找到id为{}的块".format(id))
            cls = component_or_layout_class(block_config["type"])
            block_config["props"].pop("type", None)
            block_config["props"].pop("name", None)
            style = block_config["props"].pop("style", None)
            if block_config["props"].get("root_url") is None and root_url:
                block_config["props"]["root_url"] = root_url + "/"
              # 任何组件已经处理了其初始值，因此我们在这里跳过该步骤
            block = cls(**block_config["props"], _skip_init_processing=True)
            if style and isinstance(block, components.IOComponent):
                block.style(**style)
            return block

        def iterate_over_children(children_list):
            for child_config in children_list:
                id = child_config["id"]
                block = get_block_instance(id)

                original_mapping[id] = block

                children = child_config.get("children")
                if children is not None:
                    assert isinstance(
                        block, BlockContext
                    ), f"Invalid config, Block with id {id} has children but is not a BlockContext."
                    with block:
                        iterate_over_children(children)

        derived_fields = ["types"]

        with Blocks() as blocks:
            # ID 0 should be the root Blocks component
            original_mapping[0] = Context.root_block or blocks

            iterate_over_children(config["layout"]["children"])

            first_dependency = None

            # add the event triggers
            for dependency, fn in zip(config["dependencies"], fns):
                # We used to add a "fake_event" to the config to cache examples
                # without removing it. This was causing bugs in calling gr.Interface.load
                # We fixed the issue by removing "fake_event" from the config in examples.py
                # but we still need to skip these events when loading the config to support
                # older demos
                if dependency["trigger"] == "fake_event":
                    continue
                for field in derived_fields:
                    dependency.pop(field, None)
                targets = dependency.pop("targets")
                trigger = dependency.pop("trigger")
                dependency.pop("backend_fn")
                dependency.pop("documentation", None)
                dependency["inputs"] = [
                    original_mapping[i] for i in dependency["inputs"]
                ]
                dependency["outputs"] = [
                    original_mapping[o] for o in dependency["outputs"]
                ]
                dependency.pop("status_tracker", None)
                dependency["preprocess"] = False
                dependency["postprocess"] = False

                for target in targets:
                    dependency = original_mapping[target].set_event_trigger(
                        event_name=trigger, fn=fn, **dependency
                    )[0]
                    if first_dependency is None:
                        first_dependency = dependency

            # Allows some use of Interface-specific methods with loaded Spaces
            if first_dependency and Context.root_block:
                blocks.predict = [fns[0]]
                blocks.input_components = [
                    Context.root_block.blocks[i] for i in first_dependency["inputs"]
                ]
                blocks.output_components = [
                    Context.root_block.blocks[o] for o in first_dependency["outputs"]
                ]
                blocks.__name__ = "Interface"
                blocks.api_mode = True

        return blocks

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        num_backend_fns = len([d for d in self.dependencies if d["backend_fn"]])
        repr = f"Gradio Blocks instance: {num_backend_fns} backend functions"
        repr += "\n" + "-" * len(repr)
        for d, dependency in enumerate(self.dependencies):
            if dependency["backend_fn"]:
                repr += f"\nfn_index={d}"
                repr += "\n inputs:"
                for input_id in dependency["inputs"]:
                    block = self.blocks[input_id]
                    repr += "\n |-{}".format(str(block))
                repr += "\n outputs:"
                for output_id in dependency["outputs"]:
                    block = self.blocks[output_id]
                    repr += "\n |-{}".format(str(block))
        return repr

    def render(self):
        if Context.root_block is not None:
            if self._id in Context.root_block.blocks:
                raise DuplicateBlockError(
                    f"A block with id: {self._id} has already been rendered in the current Blocks."
                )
            if not set(Context.root_block.blocks).isdisjoint(self.blocks):
                raise DuplicateBlockError(
                    "At least one block in this Blocks has already been rendered."
                )

            Context.root_block.blocks.update(self.blocks)
            Context.root_block.fns.extend(self.fns)
            dependency_offset = len(Context.root_block.dependencies)
            for i, dependency in enumerate(self.dependencies):
                api_name = dependency["api_name"]
                if api_name is not None:
                    api_name_ = utils.append_unique_suffix(
                        api_name,
                        [dep["api_name"] for dep in Context.root_block.dependencies],
                    )
                    if not (api_name == api_name_):
                        warnings.warn(
                            "api_name {} already exists, using {}".format(
                                api_name, api_name_
                            )
                        )
                        dependency["api_name"] = api_name_
                dependency["cancels"] = [
                    c + dependency_offset for c in dependency["cancels"]
                ]
                if dependency.get("trigger_after") is not None:
                    dependency["trigger_after"] += dependency_offset
                # Recreate the cancel function so that it has the latest
                # dependency fn indices. This is necessary to properly cancel
                # events in the backend
                if dependency["cancels"]:
                    updated_cancels = [
                        Context.root_block.dependencies[i]
                        for i in dependency["cancels"]
                    ]
                    new_fn = BlockFunction(
                        get_cancel_function(updated_cancels)[0],
                        [],
                        [],
                        False,
                        True,
                        False,
                    )
                    Context.root_block.fns[dependency_offset + i] = new_fn
                Context.root_block.dependencies.append(dependency)
            Context.root_block.temp_file_sets.extend(self.temp_file_sets)

        if Context.block is not None:
            Context.block.children.extend(self.children)
        return self

    def is_callable(self, fn_index: int = 0) -> bool:
        """Checks if a particular Blocks function is callable (i.e. not stateful or a generator)."""
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        if inspect.isasyncgenfunction(block_fn.fn):
            return False
        if inspect.isgeneratorfunction(block_fn.fn):
            return False
        for input_id in dependency["inputs"]:
            block = self.blocks[input_id]
            if getattr(block, "stateful", False):
                return False
        for output_id in dependency["outputs"]:
            block = self.blocks[output_id]
            if getattr(block, "stateful", False):
                return False

        return True

    def __call__(self, *inputs, fn_index: int = 0, api_name: str | None = None):
        """
        Allows Blocks objects to be called as functions. Supply the parameters to the
        function as positional arguments. To choose which function to call, use the
        fn_index parameter, which must be a keyword argument.

        Parameters:
        *inputs: the parameters to pass to the function
        fn_index: the index of the function to call (defaults to 0, which for Interfaces, is the default prediction function)
        api_name: The api_name of the dependency to call. Will take precedence over fn_index.
        """
        if api_name is not None:
            inferred_fn_index = next(
                (
                    i
                    for i, d in enumerate(self.dependencies)
                    if d.get("api_name") == api_name
                ),
                None,
            )
            if inferred_fn_index is None:
                raise InvalidApiName(f"Cannot find a function with api_name {api_name}")
            fn_index = inferred_fn_index
        if not (self.is_callable(fn_index)):
            raise ValueError(
                "This function is not callable because it is either stateful or is a generator. Please use the .launch() method instead to create an interactive user interface."
            )

        inputs = list(inputs)
        processed_inputs = self.serialize_data(fn_index, inputs)
        batch = self.dependencies[fn_index]["batch"]
        if batch:
            processed_inputs = [[inp] for inp in processed_inputs]

        outputs = utils.synchronize_async(
            self.process_api,
            fn_index=fn_index,
            inputs=processed_inputs,
            request=None,
            state={},
        )
        outputs = outputs["data"]

        if batch:
            outputs = [out[0] for out in outputs]

        processed_outputs = self.deserialize_data(fn_index, outputs)
        processed_outputs = utils.resolve_singleton(processed_outputs)

        return processed_outputs

    async def call_function(
        self,
        fn_index: int,
        processed_input: List[Any],
        iterator: Iterator[Any] | None = None,
        requests: routes.Request | List[routes.Request] | None = None,
        event_id: str | None = None,
        event_data: EventData | None = None,
    ):
        """
        Calls function with given index and preprocessed input, and measures process time.
        Parameters:
            fn_index: index of function to call
            processed_input: preprocessed input to pass to function
            iterator: iterator to use if function is a generator
            requests: requests to pass to function
            event_id: id of event in queue
            event_data: data associated with event trigger
        """
        block_fn = self.fns[fn_index]
        assert block_fn.fn, f"function with index {fn_index} not defined."
        is_generating = False

        if block_fn.inputs_as_dict:
            processed_input = [
                {
                    input_component: data
                    for input_component, data in zip(block_fn.inputs, processed_input)
                }
            ]

        if isinstance(requests, list):
            request = requests[0]
        else:
            request = requests
        processed_input, progress_index, _ = special_args(
            block_fn.fn, processed_input, request, event_data
        )
        progress_tracker = (
            processed_input[progress_index] if progress_index is not None else None
        )

        start = time.time()

        if iterator is None:  # If not a generator function that has already run
            if progress_tracker is not None and progress_index is not None:
                progress_tracker, fn = create_tracker(
                    self, event_id, block_fn.fn, progress_tracker.track_tqdm
                )
                processed_input[progress_index] = progress_tracker
            else:
                fn = block_fn.fn

            if inspect.iscoroutinefunction(fn):
                prediction = await fn(*processed_input)
            else:
                prediction = await anyio.to_thread.run_sync(
                    fn, *processed_input, limiter=self.limiter
                )
        else:
            prediction = None

        if inspect.isasyncgenfunction(block_fn.fn):
            raise ValueError("Gradio does not support async generators.")
        if inspect.isgeneratorfunction(block_fn.fn):
            if not self.enable_queue:
                raise ValueError("Need to enable queue to use generators.")
            try:
                if iterator is None:
                    iterator = prediction
                prediction = await anyio.to_thread.run_sync(
                    utils.async_iteration, iterator, limiter=self.limiter
                )
                is_generating = True
            except StopAsyncIteration:
                n_outputs = len(self.dependencies[fn_index].get("outputs"))
                prediction = (
                    components._Keywords.FINISHED_ITERATING
                    if n_outputs == 1
                    else (components._Keywords.FINISHED_ITERATING,) * n_outputs
                )
                iterator = None

        duration = time.time() - start

        return {
            "prediction": prediction,
            "duration": duration,
            "is_generating": is_generating,
            "iterator": iterator,
        }

    def serialize_data(self, fn_index: int, inputs: List[Any]) -> List[Any]:
        dependency = self.dependencies[fn_index]
        processed_input = []

        for i, input_id in enumerate(dependency["inputs"]):
            block = self.blocks[input_id]
            assert isinstance(
                block, components.IOComponent
            ), f"{block.__class__} Component with id {input_id} not a valid input component."
            serialized_input = block.serialize(inputs[i])
            processed_input.append(serialized_input)

        return processed_input

    def deserialize_data(self, fn_index: int, outputs: List[Any]) -> List[Any]:
        dependency = self.dependencies[fn_index]
        predictions = []

        for o, output_id in enumerate(dependency["outputs"]):
            block = self.blocks[output_id]
            assert isinstance(
                block, components.IOComponent
            ), f"{block.__class__} Component with id {output_id} not a valid output component."
            deserialized = block.deserialize(outputs[o], root_url=block.root_url)
            predictions.append(deserialized)

        return predictions

    def preprocess_data(self, fn_index: int, inputs: List[Any], state: Dict[int, Any]):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        if block_fn.preprocess:
            processed_input = []
            for i, input_id in enumerate(dependency["inputs"]):
                block = self.blocks[input_id]
                assert isinstance(
                    block, components.Component
                ), f"{block.__class__} Component with id {input_id} not a valid input component."
                if getattr(block, "stateful", False):
                    processed_input.append(state.get(input_id))
                else:
                    processed_input.append(block.preprocess(inputs[i]))
        else:
            processed_input = inputs
        return processed_input

    def postprocess_data(
        self, fn_index: int, predictions: List | Dict, state: Dict[int, Any]
    ):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]
        batch = dependency["batch"]

        if type(predictions) is dict and len(predictions) > 0:
            predictions = convert_component_dict_to_list(
                dependency["outputs"], predictions
            )

        if len(dependency["outputs"]) == 1 and not (batch):
            predictions = [
                predictions,
            ]

        output = []
        for i, output_id in enumerate(dependency["outputs"]):
            try:
                if predictions[i] is components._Keywords.FINISHED_ITERATING:
                    output.append(None)
                    continue
            except (IndexError, KeyError):
                raise ValueError(
                    f"Number of output components does not match number of values returned from from function {block_fn.name}"
                )
            block = self.blocks[output_id]
            if getattr(block, "stateful", False):
                if not utils.is_update(predictions[i]):
                    state[output_id] = predictions[i]
                output.append(None)
            else:
                prediction_value = predictions[i]
                if utils.is_update(prediction_value):
                    assert isinstance(prediction_value, dict)
                    prediction_value = postprocess_update_dict(
                        block=block,
                        update_dict=prediction_value,
                        postprocess=block_fn.postprocess,
                    )
                elif block_fn.postprocess:
                    assert isinstance(
                        block, components.Component
                    ), f"{block.__class__} Component with id {output_id} not a valid output component."
                    prediction_value = block.postprocess(prediction_value)
                output.append(prediction_value)

        return output

    async def process_api(
        self,
        fn_index: int,
        inputs: List[Any],
        state: Dict[int, Any],
        request: routes.Request | List[routes.Request] | None = None,
        iterators: Dict[int, Any] | None = None,
        event_id: str | None = None,
        event_data: EventData | None = None,
    ) -> Dict[str, Any]:
        """
        处理来自前端的API调用。首先预处理数据，然后运行相关函数，最后对输出进行后处理。

    参数：
        fn_index：要运行的函数索引。
        inputs：从前端接收到的输入数据
        username：如果设置了身份验证，则为用户名称（未使用）
        state：为会话存储的数据来自有状态组件（键是输入块id）
        iterators：每个生成器函数的正在进行的迭代器（键为函数索引）
        event_id：触发此API调用的事件ID
        event_data：与事件触发本身相关联的数据
    返回：无
        """
        block_fn = self.fns[fn_index]
        batch = self.dependencies[fn_index]["batch"]

        if batch:
            max_batch_size = self.dependencies[fn_index]["max_batch_size"]
            batch_sizes = [len(inp) for inp in inputs]
            batch_size = batch_sizes[0]
            if inspect.isasyncgenfunction(block_fn.fn) or inspect.isgeneratorfunction(
                block_fn.fn
            ):
                raise ValueError("Gradio does not support generators in batch mode.")
            if not all(x == batch_size for x in batch_sizes):
                raise ValueError(
                    f"All inputs to a batch function must have the same length but instead have sizes: {batch_sizes}."
                )
            if batch_size > max_batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) exceeds the max_batch_size for this function ({max_batch_size})"
                )

            inputs = [
                self.preprocess_data(fn_index, list(i), state) for i in zip(*inputs)
            ]
            result = await self.call_function(
                fn_index, list(zip(*inputs)), None, request, event_id, event_data
            )
            preds = result["prediction"]
            data = [
                self.postprocess_data(fn_index, list(o), state) for o in zip(*preds)
            ]
            data = list(zip(*data))
            is_generating, iterator = None, None
        else:
            inputs = self.preprocess_data(fn_index, inputs, state)
            iterator = iterators.get(fn_index, None) if iterators else None
            result = await self.call_function(
                fn_index, inputs, iterator, request, event_id, event_data
            )
            data = self.postprocess_data(fn_index, result["prediction"], state)
            is_generating, iterator = result["is_generating"], result["iterator"]

        block_fn.total_runtime += result["duration"]
        block_fn.total_runs += 1

        return {
            "data": data,
            "is_generating": is_generating,
            "iterator": iterator,
            "duration": result["duration"],
            "average_duration": block_fn.total_runtime / block_fn.total_runs,
        }

    async def create_limiter(self):
        self.limiter = (
            None
            if self.max_threads == 40
            else CapacityLimiter(total_tokens=self.max_threads)
        )

    def get_config(self):
        return {"type": "column"}

    def get_config_file(self):
        config = {
            "version": routes.VERSION,
            "mode": self.mode,
            "dev_mode": self.dev_mode,
            "analytics_enabled": self.analytics_enabled,
            "components": [],
            "css": self.css,
            "title": self.title or "Gradio",
            "is_space": self.is_space,
            "enable_queue": getattr(self, "enable_queue", False),  # launch attributes
            "show_error": getattr(self, "show_error", False),
            "show_api": self.show_api,
            "is_colab": utils.colab_check(),
            "stylesheets": self.stylesheets,
            "root": self.root,
        }

        def getLayout(block):
            if not isinstance(block, BlockContext):
                return {"id": block._id}
            children_layout = []
            for child in block.children:
                children_layout.append(getLayout(child))
            return {"id": block._id, "children": children_layout}

        config["layout"] = getLayout(self)

        for _id, block in self.blocks.items():
            config["components"].append(
                {
                    "id": _id,
                    "type": (block.get_block_name()),
                    "props": utils.delete_none(block.get_config())
                    if hasattr(block, "get_config")
                    else {},
                }
            )
        config["dependencies"] = self.dependencies
        return config

    def __enter__(self):
        if Context.block is None:
            Context.root_block = self
        self.parent = Context.block
        Context.block = self
        return self

    def __exit__(self, *args):
        super().fill_expected_parents()
        Context.block = self.parent
        # Configure the load events before root_block is reset
        self.attach_load_events()
        if self.parent is None:
            Context.root_block = None
        else:
            self.parent.children.extend(self.children)
        self.config = self.get_config_file()
        self.app = routes.App.create_app(self)
        self.progress_tracking = any(block_fn.tracks_progress for block_fn in self.fns)

    @class_or_instancemethod
    def load(
        self_or_cls,
        fn: Callable | None = None,
        inputs: List[Component] | None = None,
        outputs: List[Component] | None = None,
        api_name: str | None = None,
        scroll_to_output: bool = False,
        show_progress: bool = True,
        queue=None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        every: float | None = None,
        _js: str | None = None,
        *,
        name: str | None = None,
        src: str | None = None,
        api_key: str | None = None,
        alias: str | None = None,
        **kwargs,
    ) -> Blocks | Dict[str, Any] | None:
        """
       出于反向兼容性原因，这既是一个类方法，也是一个实例方法，它们两个令人困惑的地方是它们执行完全不同的操作。

    类方法：从Hugging Face Spaces repo加载演示文稿并在本地创建它并返回块实例。等效于gradio.Interface.load（）


    实例方法：添加事件，在演示文稿在浏览器中加载时立即运行。以下是示例用法。
    参数：
        name: 类方法 - 模型名称（例如“gpt2”或“facebook / bart-base”）或空间名称（例如“flax-community / spanish-gpt2”），可以包括src作为前缀（例如“models / facebook / bart-base”）
        src: 类方法 - 型号来源：“models”或“spaces”（如果source作为前缀提供在“name”中，请留空）
        api_key: 类方法 - 用于加载私有Hugging Face Hub模型或空间的可选访问令牌。在此找到您的令牌：https://huggingface.co/settings/tokens
        alias: 类方法 - 可选字符串，用作加载的模型名称而不是默认名称（仅适用于加载运行Gradio 2.x的Space）
        fn: 实例方法 - 要包装界面的函数。通常是机器学习模型的预测函数。函数的每个参数对应于一个输入组件，并且函数应返回单个值或元组值，其中元组中的每个元素对应于一个输出组件。
        inputs: 实例方法 - 要用作输入的gradio.components列表。如果函数不接受任何输入，则应该是一个空列表。
        outputs: 实例方法 - 要用作输入的gradio.components列表。如果函数不返回任何输出，则应该是一个空列表。
        api_name: 实例方法 - 定义此参数将在API文档中公开端点
        scroll_to_output: 实例方法 - 如果为True，则会滚动到输出组件以完成
        show_progress: 实例方法 - 如果为True，则会在等待期间显示进度动画
        queue: 实例方法 - 如果为True，则会将请求放置在队列中（如果队列存在）
        batch: 实例方法 - 如果为True，则函数应处理一批输入，这意味着它应接受每个参数的输入值列表。列表应具有相等的长度（并且最长为'max_batch_size'）。然后*必须*返回元组列表（即使只有1个输出组件），元组中的每个列表都对应于一个输出组件。
        max_batch_size: 实例方法 - 如果从队列调用此方法，则最多将输入批处理在一起的数量（仅当batch=True时相关）
        preprocess: 实例方法 - 如果为False，则在运行'fn'之前不会运行组件数据的预处理（例如，如果使用'Image'组件调用此方法，则将其保留为base64字符串）。
        postprocess: 实例方法 - 如果为False，则在将'fn'输出返回到浏览器之前不会对组件数据进行后处理。
        every: 实例方法 - 每'every'秒运行此事件。以秒为单位解释。必须启用队列。
    示例：
        import gradio as gr
        import datetime
        with gr.Blocks() as demo:
            def get_time():
                return datetime.datetime.now().time()
            dt = gr.Textbox(label="Current time")
            demo.load(get_time, inputs=None, outputs=dt)
        demo.launch()
        """
        # _js: Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
        if isinstance(self_or_cls, type):
            if name is None:
                raise ValueError(
                    "Blocks.load() requires passing parameters as keyword arguments"
                )
            return external.load_blocks_from_repo(name, src, api_key, alias, **kwargs)
        else:
            return self_or_cls.set_event_trigger(
                event_name="load",
                fn=fn,
                inputs=inputs,
                outputs=outputs,
                api_name=api_name,
                preprocess=preprocess,
                postprocess=postprocess,
                scroll_to_output=scroll_to_output,
                show_progress=show_progress,
                js=_js,
                queue=queue,
                batch=batch,
                max_batch_size=max_batch_size,
                every=every,
                no_target=True,
            )[0]

    def clear(self):
        """Resets the layout of the Blocks object."""
        self.blocks = {}
        self.fns = []
        self.dependencies = []
        self.children = []
        return self

    @document()
    def queue(
        self,
        concurrency_count: int = 1,
        status_update_rate: float | Literal["auto"] = "auto",
        client_position_to_load_data: int | None = None,
        default_enabled: bool | None = None,
        api_open: bool = True,
        max_size: int | None = None,
    ):
        """
       你可以通过创建一个队列来控制处理请求的速率。这将允许你设置一次处理的请求数量，并让用户知道他们在队列中的位置。
参数:
concurrency_count: 并发处理请求的工作线程数量。增加此数字将增加处理请求的速度，但也会增加队列的内存使用量。
status_update_rate: 如果为“auto”，Queue将在每个作业完成时向所有客户端发送状态估计值。否则，Queue将按照此参数设置的时间间隔定期发送状态，单位为秒。
client_position_to_load_data: 已弃用。该参数已被弃用，不再起作用。
default_enabled: 已弃用。该参数已被弃用，不再起作用。
api_open: 如果为True，则后端的REST路由将是开放的，允许直接对这些端点进行请求而跳过队列。
max_size: 队列在任何时刻存储的事件的最大数量。如果队列已满，则不会添加新事件，并且用户将收到一条消息，说明队列已满。如果为None，则队列大小将是无限的。
示例：（Blocks）
with gr.Blocks() as demo:
button = gr.Button(label="Generate Image")
button.click(fn=image_generator, inputs=gr.Textbox(), outputs=gr.Image())
demo.queue(concurrency_count=3)
demo.launch()
示例：(Interface)
demo = gr.Interface(image_generator, gr.Textbox(), gr.Image())
demo.queue(concurrency_count=3)
demo.launch()
        """
        if default_enabled is not None:
            warnings.warn(
                "The default_enabled parameter of queue has no effect and will be removed "
                "in a future version of gradio."
            )
        self.enable_queue = True
        self.api_open = api_open
        if client_position_to_load_data is not None:
            warnings.warn("The client_position_to_load_data parameter is deprecated.")
        self._queue = queueing.Queue(
            live_updates=status_update_rate == "auto",
            concurrency_count=concurrency_count,
            update_intervals=status_update_rate if status_update_rate != "auto" else 1,
            max_size=max_size,
            blocks_dependencies=self.dependencies,
        )
        self.config = self.get_config_file()
        self.app = routes.App.create_app(self)
        return self

    def launch(
        self,
        inline: bool | None = None,
        inbrowser: bool = False,
        share: bool | None = None,
        debug: bool = False,
        enable_queue: bool | None = None,
        max_threads: int = 40,
        auth: Callable | Tuple[str, str] | List[Tuple[str, str]] | None = None,
        auth_message: str | None = None,
        prevent_thread_lock: bool = False,
        show_error: bool = False,
        server_name: str | None = None,
        server_port: int | None = None,
        show_tips: bool = False,
        height: int = 500,
        width: int | str = "100%",
        encrypt: bool | None = None,
        favicon_path: str | None = None,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_keyfile_password: str | None = None,
        quiet: bool = False,
        show_api: bool = True,
        file_directories: List[str] | None = None,
        _frontend: bool = True,
    ) -> Tuple[FastAPI, str, str]:
        """
       启动一个简单的Web服务器，用于提供演示。还可以通过设置share=True来创建一个供任何人从其浏览器访问演示的公共链接。

参数:
inline: 是否在界面中内联显示iframe。在python笔记本中默认为True；否则为False。
inbrowser: 是否在默认浏览器中自动启动界面的新选项卡。
share: 是否为界面创建可公开共享的链接。创建SSH隧道以使您的UI可以从任何地方访问。如果未提供，则每次都会默认设置为False，除非在Google Colab中运行。当本地主机不可访问（例如Google Colab）时，不支持将share=False设置为。
debug: 如果为True，则阻止主线程运行。在Google Colab中运行时，这是需要打印单元格输出中的错误的。
auth: 如果提供，则需要用户名和密码（或用户名-密码元组列表）才能访问界面。也可以提供函数，该函数接受用户名和密码并返回True，如果登录有效。
auth_message: 如果提供，则在登录页面上提供HTML消息。
prevent_thread_lock: 如果为True，则在服务器运行时，界面将阻止主线程。
show_error: 如果为True，则在警报模态窗口中显示界面中的任何错误，并打印在浏览器控制台日志中
server_port: 将在此端口（如果有）上启动gradio应用程序。可以通过环境变量GRADIO_SERVER_PORT设置。如果为None，则将从7860开始搜索可用端口。
server_name: 要使应用程序在本地网络上可访问，请将其设置为“0.0.0.0”。可以通过环境变量GRADIO_SERVER_NAME设置。如果为None，则使用“127.0.0.1”。
show_tips: 如果为True，则会偶尔显示有关新Gradio功能的提示
enable_queue: 已弃用（改用.queue()方法）。如果为True，则推断请求将通过队列而不是并行线程进行服务。对于较长的推理时间（> 1min），需要此选项以防止超时。在HuggingFace Spaces中的默认选项为True。其他地方的默认选项为False。
max_threads: Gradio应用程序可以并行生成的总线程数的最大值。默认继承自starlette库（当前为40）。无论是否启用队列，都适用。但是，如果启用排队，则将增加此参数至少为队列的concurrency_count。
width: 包含界面的iframe元素的像素宽度（如果inline=True）
height: 包含界面的iframe元素的像素高度（如果inline=True）
encrypt: 已弃用。没有效果。
favicon_path: 如果提供了一个文件路径（.png、.gif或.ico文件），则将其用作网页的favicon。
ssl_keyfile: 如果提供了文件路径，则将其用作创建在https上运行的本地服务器的私钥文件。
ssl_certfile: 如果提供了文件路径，则将用其作为https的已签名证书。如果提供了ssl_keyfile，则需要提供此项。
ssl_keyfile_password: 如果提供了密码，则将其与https证书一起使用。
quiet: 如果为True，则抑制大多数打印语句。
show_api: 如果为True，则在应用程序底部显示API文档。默认值为True。如果启用了队列，则.queue()的api_open参数将确定是否显示api文档，而与show_api的值无关。
file_directories: 允许Gradio从中提供文件的目录列表（除了包含gradio python文件的目录）。必须是绝对路径。警告：这些目录或其子级中的任何文件都可能可被您应用程序的所有用户访问到。
        """
        self.dev_mode = False
        if (
            auth
            and not callable(auth)
            and not isinstance(auth[0], tuple)
            and not isinstance(auth[0], list)
        ):
            self.auth = [auth]
        else:
            self.auth = auth
        self.auth_message = auth_message
        self.show_tips = show_tips
        self.show_error = show_error
        self.height = height
        self.width = width
        self.favicon_path = favicon_path

        if enable_queue is not None:
            self.enable_queue = enable_queue
            warnings.warn(
                "The `enable_queue` parameter has been deprecated. Please use the `.queue()` method instead.",
                DeprecationWarning,
            )
        if encrypt is not None:
            warnings.warn(
                "The `encrypt` parameter has been deprecated and has no effect.",
                DeprecationWarning,
            )

        if self.is_space:
            self.enable_queue = self.enable_queue is not False
        else:
            self.enable_queue = self.enable_queue is True
        if self.enable_queue and not hasattr(self, "_queue"):
            self.queue()
        self.show_api = self.api_open if self.enable_queue else show_api

        self.file_directories = file_directories if file_directories is not None else []
        if not isinstance(self.file_directories, list):
            raise ValueError("file_directories must be a list of directories.")

        if not self.enable_queue and self.progress_tracking:
            raise ValueError("Progress tracking requires queuing to be enabled.")

        for dep in self.dependencies:
            for i in dep["cancels"]:
                if not self.queue_enabled_for_fn(i):
                    raise ValueError(
                        "In order to cancel an event, the queue for that event must be enabled! "
                        "You may get this error by either 1) passing a function that uses the yield keyword "
                        "into an interface without enabling the queue or 2) defining an event that cancels "
                        "another event without enabling the queue. Both can be solved by calling .queue() "
                        "before .launch()"
                    )
            if dep["batch"] and (
                dep["queue"] is False
                or (dep["queue"] is None and not self.enable_queue)
            ):
                raise ValueError("In order to use batching, the queue must be enabled.")

        self.config = self.get_config_file()
        self.max_threads = max(
            self._queue.max_thread_count if self.enable_queue else 0, max_threads
        )

        if self.is_running:
            assert isinstance(
                self.local_url, str
            ), f"Invalid local_url: {self.local_url}"
            if not (quiet):
                print(
                    "Rerunning server... use `close()` to stop if you need to change `launch()` parameters.\n----"
                )
        else:
            server_name, server_port, local_url, app, server = networking.start_server(
                self,
                server_name,
                server_port,
                ssl_keyfile,
                ssl_certfile,
                ssl_keyfile_password,
            )
            self.server_name = server_name
            self.local_url = local_url
            self.server_port = server_port
            self.server_app = app
            self.server = server
            self.is_running = True
            self.is_colab = utils.colab_check()
            self.is_kaggle = utils.kaggle_check()
            self.is_sagemaker = utils.sagemaker_check()

            self.protocol = (
                "https"
                if self.local_url.startswith("https") or self.is_colab
                else "http"
            )

            if self.enable_queue:
                self._queue.set_url(self.local_url)

            # Cannot run async functions in background other than app's scope.
            # Workaround by triggering the app endpoint
            requests.get(f"{self.local_url}startup-events")

        utils.launch_counter()

        if share is None:
            if self.is_colab and self.enable_queue:
                if not quiet:
                    print(
                        "Setting queue=True in a Colab notebook requires sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            elif self.is_kaggle:
                if not quiet:
                    print(
                        "Kaggle notebooks require sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            elif self.is_sagemaker:
                if not quiet:
                    print(
                        "Sagemaker notebooks may require sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            else:
                self.share = False
        else:
            self.share = share

        # If running in a colab or not able to access localhost,
        # a shareable link must be created.
        if _frontend and (not networking.url_ok(self.local_url)) and (not self.share):
            raise ValueError(
                "When localhost is not accessible, a shareable link must be created. Please set share=True."
            )

        if self.is_colab:
            if not quiet:
                if debug:
                    print(strings.en["COLAB_DEBUG_TRUE"])
                else:
                    print(strings.en["COLAB_DEBUG_FALSE"])
                if not self.share:
                    print(strings.en["COLAB_WARNING"].format(self.server_port))
            if self.enable_queue and not self.share:
                raise ValueError(
                    "When using queueing in Colab, a shareable link must be created. Please set share=True."
                )
        else:
            if not self.share:
              print(f'Running on local URL: https://{self.server_name}') 


        if self.share:
            if self.is_space:
                raise RuntimeError("Share is not supported when you are in Spaces")
            try:
                if self.share_url is None:
                    self.share_url = networking.setup_tunnel(
                        self.server_name, self.server_port, self.share_token
                    )
                print(strings.en["SHARE_LINK_DISPLAY"].format(self.share_url))
                if not (quiet):
                    print('[32m\u2714 Connected')
            except (RuntimeError, requests.exceptions.ConnectionError):
                if self.analytics_enabled:
                    utils.error_analytics("Not able to set up tunnel")
                self.share_url = None
                self.share = False
                print(strings.en["COULD_NOT_GET_SHARE_LINK"])
        else:
            if not (quiet):
                print('[32m\u2714 Connected')
            self.share_url = None

        if inbrowser:
            link = self.share_url if self.share and self.share_url else self.local_url
            webbrowser.open(link)

        # Check if running in a Python notebook in which case, display inline
        if inline is None:
            inline = utils.ipython_check() and (self.auth is None)
        if inline:
            if self.auth is not None:
                print(
                    "Warning: authentication is not supported inline. Please"
                    "click the link to access the interface in a new tab."
                )
            try:
                from IPython.display import HTML, Javascript, display  # type: ignore

                if self.share and self.share_url:
                    while not networking.url_ok(self.share_url):
                        time.sleep(0.25)
                    display(
                        HTML(
                            f'<div><iframe src="{self.share_url}" width="{self.width}" height="{self.height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
                        )
                    )
                elif self.is_colab:
                    # modified from /usr/local/lib/python3.7/dist-packages/google/colab/output/_util.py within Colab environment
                    code = """(async (port, path, width, height, cache, element) => {
                        if (!google.colab.kernel.accessAllowed && !cache) {
                            return;
                        }
                        element.appendChild(document.createTextNode(''));
                        const url = await google.colab.kernel.proxyPort(port, {cache});

                        const external_link = document.createElement('div');
                        external_link.innerHTML = `
                            <div style="font-family: monospace; margin-bottom: 0.5rem">
                                Running on <a href=${new URL(path, url).toString()} target="_blank">
                                    https://localhost:${port}${path}
                                </a>
                            </div>
                        `;
                        element.appendChild(external_link);

                        const iframe = document.createElement('iframe');
                        iframe.src = new URL(path, url).toString();
                        iframe.height = height;
                        iframe.allow = "autoplay; camera; microphone; clipboard-read; clipboard-write;"
                        iframe.width = width;
                        iframe.style.border = 0;
                        element.appendChild(iframe);
                    })""" + "({port}, {path}, {width}, {height}, {cache}, window.element)".format(
                        port=json.dumps(self.server_port),
                        path=json.dumps("/"),
                        width=json.dumps(self.width),
                        height=json.dumps(self.height),
                        cache=json.dumps(False),
                    )

                    display(Javascript(code))
                else:
                    display(
                        HTML(
                            f'<div><iframe src="{self.local_url}" width="{self.width}" height="{self.height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
                        )
                    )
            except ImportError:
                pass

        if getattr(self, "analytics_enabled", False):
            data = {
                "launch_method": "browser" if inbrowser else "inline",
                "is_google_colab": self.is_colab,
                "is_sharing_on": self.share,
                "share_url": self.share_url,
                "enable_queue": self.enable_queue,
                "show_tips": self.show_tips,
                "server_name": server_name,
                "server_port": server_port,
                "is_spaces": self.is_space,
                "mode": self.mode,
            }
            utils.launch_analytics(data)
            utils.launched_telemetry(self, data)

        utils.show_tip(self)

        # Block main thread if debug==True
        if debug or int(os.getenv("GRADIO_DEBUG", 0)) == 1:
            self.block_thread()
        # Block main thread if running in a script to stop script from exiting
        is_in_interactive_mode = bool(getattr(sys, "ps1", sys.flags.interactive))

        if not prevent_thread_lock and not is_in_interactive_mode:
            self.block_thread()

        return TupleNoPrint((self.server_app, self.local_url, self.share_url))

    def integrate(
        self,
        comet_ml: comet_ml.Experiment | None = None,
        wandb: ModuleType | None = None,
        mlflow: ModuleType | None = None,
    ) -> None:
        """
        A catch-all method for integrating with other libraries. This method should be run after launch()
        Parameters:
            comet_ml: If a comet_ml Experiment object is provided, will integrate with the experiment and appear on Comet dashboard
            wandb: If the wandb module is provided, will integrate with it and appear on WandB dashboard
            mlflow: If the mlflow module  is provided, will integrate with the experiment and appear on ML Flow dashboard
        """
        analytics_integration = ""
        if comet_ml is not None:
            analytics_integration = "CometML"
            comet_ml.log_other("Created from", "Gradio")
            if self.share_url is not None:
                comet_ml.log_text("gradio: " + self.share_url)
                comet_ml.end()
            elif self.local_url:
                comet_ml.log_text("gradio: " + self.local_url)
                comet_ml.end()
            else:
                raise ValueError("Please run `launch()` first.")
        if wandb is not None:
            analytics_integration = "WandB"
            if self.share_url is not None:
                wandb.log(
                    {
                        "Gradio panel": wandb.Html(
                            '<iframe src="'
                            + self.share_url
                            + '" width="'
                            + str(self.width)
                            + '" height="'
                            + str(self.height)
                            + '" frameBorder="0"></iframe>'
                        )
                    }
                )
            else:
                print(
                    "The WandB integration requires you to "
                    "`launch(share=True)` first."
                )
        if mlflow is not None:
            analytics_integration = "MLFlow"
            if self.share_url is not None:
                mlflow.log_param("Gradio Interface Share Link", self.share_url)
            else:
                mlflow.log_param("Gradio Interface Local Link", self.local_url)
        if self.analytics_enabled and analytics_integration:
            data = {"integration": analytics_integration}
            utils.integration_analytics(data)

    def close(self, verbose: bool = True) -> None:
        """
        Closes the Interface that was launched and frees the port.
        """
        try:
            if self.enable_queue:
                self._queue.close()
            self.server.close()
            self.is_running = False
            # So that the startup events (starting the queue)
            # happen the next time the app is launched
            self.app.startup_events_triggered = False
            if verbose:
                print("Closing server running on port: {}".format(self.server_port))
        except (AttributeError, OSError):  # can't close if not running
            pass

    def block_thread(
        self,
    ) -> None:
        """Block main thread until interrupted by user."""
        try:
            while True:
                time.sleep(0.1)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            self.server.close()
            for tunnel in CURRENT_TUNNELS:
                tunnel.kill()

    def attach_load_events(self):
        """Add a load event for every component whose initial value should be randomized."""
        if Context.root_block:
            for component in Context.root_block.blocks.values():
                if (
                    isinstance(component, components.IOComponent)
                    and component.load_event_to_attach
                ):
                    load_fn, every = component.load_event_to_attach
                    # Use set_event_trigger to avoid ambiguity between load class/instance method
                    dep = self.set_event_trigger(
                        "load",
                        load_fn,
                        None,
                        component,
                        no_target=True,
                        # If every is None, for sure skip the queue
                        # else, let the enable_queue parameter take precedence
                        # this will raise a nice error message is every is used
                        # without queue
                        queue=False if every is None else None,
                        every=every,
                    )[0]
                    component.load_event = dep

    def startup_events(self):
        """Events that should be run when the app containing this block starts up."""

        if self.enable_queue:
            utils.run_coro_in_background(self._queue.start, (self.progress_tracking,))
            # So that processing can resume in case the queue was stopped
            self._queue.stopped = False
        utils.run_coro_in_background(self.create_limiter)

    def queue_enabled_for_fn(self, fn_index: int):
        if self.dependencies[fn_index]["queue"] is None:
            return self.enable_queue
        return self.dependencies[fn_index]["queue"]
