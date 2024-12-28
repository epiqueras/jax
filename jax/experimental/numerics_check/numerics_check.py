# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Concatenate, ParamSpec, Protocol, TypeVar

import jax.numpy as jnp
from jax._src import (
  api,
  api_util,
  core,
  custom_derivatives,
  dtypes,
  source_info_util,
  traceback_util,
  tree_util,
  typing,
  util,
)
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax

zip = util.safe_zip


Val = Any


# Rules


class _NumericsCheckRule(Protocol):
  def __call__(
    self,
    trace: "NumericsCheckTrace",
    in_metrics: tuple[typing.Array, ...],
    out_metric: typing.Array,
    *args: Val,
    **params: Val,
  ) -> tuple[Val, ...]: ...


_numerics_checks: dict[core.Primitive, _NumericsCheckRule] = {}


def register_numerics_check(prim: core.Primitive):
  def register(rule: _NumericsCheckRule):
    _numerics_checks[prim] = rule
    return rule

  return register


# Default Rule


# Cast all float32 args to bfloat16.
def _maybe_cast_bf16(val: Val) -> Val:
  if isinstance(val, typing.Array) and val.dtype == jnp.float32:
    val = lax.reduce_precision(val, exponent_bits=8, mantissa_bits=7)
    val = val.astype(dtypes.bfloat16)
    return val
  return val


@lru_cache
def _make_default_numerics_check(primitive: core.Primitive) -> _NumericsCheckRule:
  @lru_cache
  def make_default_numerics_check_with_kwargs(**params: Val) -> Val:
    @custom_derivatives.custom_vjp
    def default_numerics_check(
      in_metrics: tuple[typing.Array, ...], out_metric: typing.Array, *args: Val
    ) -> Val:
      del in_metrics, out_metric
      return primitive.bind(*args, **params)

    def default_numerics_fwd(
      in_metrics: tuple[typing.Array, ...], out_metric: typing.Array, *args: Val
    ):
      del out_metric
      out, f_vjp = api.vjp(lambda *args: default_numerics_check(*args, **params), args)
      low_precision_out = default_numerics_check(
        *tuple(map(_maybe_cast_bf16, args)), **params
      )
      delta = out - low_precision_out
      return out, (f_vjp, delta, in_metrics)

    def default_numerics_bwd(
      res: tuple[Callable, Val, tuple[typing.Array, ...]], g: Val
    ):
      f_vjp, delta, in_metrics = res
      out_metric = jnp.sum(g * delta)
      grads = f_vjp(g)
      low_precision_grads = f_vjp(_maybe_cast_bf16(g))
      in_metrics = tuple(
        jnp.mean(jnp.square(grad - low_precision_grad))
        for grad, low_precision_grad in zip(grads, low_precision_grads)
      )
      return (in_metrics, out_metric, *grads)

    default_numerics_check.defvjp(default_numerics_fwd, default_numerics_bwd)
    return default_numerics_check

  def default_numerics_check(
    trace: "NumericsCheckTrace",
    in_metrics: tuple[typing.Array, ...],
    out_metric: typing.Array,
    *args: Val,
    **params: Val,
  ) -> Val:
    del trace
    return make_default_numerics_check_with_kwargs(**params)(
      in_metrics, out_metric, *args
    )

  return default_numerics_check


# Trace


class NumericsCheckTracer(core.Tracer):
  _trace: "NumericsCheckTrace"
  val: Val

  def __init__(self, trace, val):
    self._trace = trace
    self.val = val

  @property
  def aval(self) -> core.AbstractValue:
    return core.get_aval(self.val)

  def to_concrete_value(self) -> Val:
    return core.to_concrete_value(self.val)


MetricsSource = dict[source_info_util.SourceInfo, int]
Metrics = dict[
  source_info_util.SourceInfo, tuple[tuple[typing.Array, ...], typing.Array]
]


class NumericsCheckTrace(core.Trace[NumericsCheckTracer]):
  parent_trace: core.Trace
  tag: core.TraceTag
  metric_sources: MetricsSource
  metrics: Metrics

  def __init__(self, parent_trace, tag, metrics: Metrics):
    self.parent_trace = parent_trace
    self.tag = tag
    self.metric_sources = {}
    self.metrics = metrics

  def to_val(self, val: Val | NumericsCheckTracer) -> Val:
    if isinstance(val, NumericsCheckTracer) and val._trace.tag is self.tag:
      return val.val
    else:
      return val

  @staticmethod
  def make_metric() -> typing.Array:
    return jnp.zeros((), dtype=jnp.float32)

  def process_primitive(
    self, primitive: core.Primitive, tracers: tuple[Val, ...], params: dict[str, Val]
  ) -> Val:
    rule = _numerics_checks.get(primitive, None)
    if rule is None:
      rule = _make_default_numerics_check(primitive)
    in_vals = tuple(map(self.to_val, tracers))
    current_source_info = source_info_util.current()
    metrics = self.metrics.get(current_source_info, ((None,) * len(in_vals), None))
    in_metrics = tuple(
      NumericsCheckTrace.make_metric() if metric is None else metric
      for metric in metrics[0]
    )
    out_metric = NumericsCheckTrace.make_metric() if metrics[1] is None else metrics[1]
    self.metric_sources[current_source_info] = len(in_metrics)
    with core.set_current_trace(self.parent_trace):
      out_vals = rule(self, in_metrics, out_metric, *in_vals, **params)
    if primitive.multiple_results:
      out_tracers = tuple(map(partial(NumericsCheckTracer, self), out_vals))
      return out_tracers
    else:
      out_tracer = NumericsCheckTracer(self, out_vals)
      return out_tracer


# Transformation


P = ParamSpec("P")
R = TypeVar("R")


@lu.transformation_with_aux2
def numerics_check_subtrace(
  f: Callable,
  store: lu.Store,
  tag: core.TraceTag,
  metrics: Metrics,
  *args_flat: Val,
) -> tuple[Val, ...]:
  with core.take_current_trace() as parent_trace:
    trace = NumericsCheckTrace(parent_trace, tag, metrics)
    in_tracers = tuple(map(partial(NumericsCheckTracer, trace), args_flat))
    with core.set_current_trace(trace):
      out_tracers = f(*in_tracers)
    out = tuple(map(trace.to_val, out_tracers))
    store.store(dict(trace=trace))
  return out


@lu.transformation2
def numerics_check_trace(
  f: Callable[Concatenate[core.TraceTag, Metrics, ...], R],
  subtrace_thunk: Callable,
  metrics: Metrics,
  *args_flat: Val,
) -> R:
  tag = core.TraceTag()
  with source_info_util.transform_name_stack("numerics_check"):
    out = f(tag, metrics, *args_flat)
  trace = subtrace_thunk()["trace"]
  with core.ensure_no_leaks(trace):
    del trace
  return out


def numerics_check(
  fun: Callable[P, R],
) -> tuple[
  Callable[Concatenate[Metrics, P], Val],
  Callable[P, MetricsSource],
]:
  api_util.check_callable(fun)
  docstr = "Takes similar arguments as {fun} but adds additional arrays in which numerical sensitivities are deposited."
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  @util.wraps(fun, docstr=docstr)
  @traceback_util.api_boundary
  def numerics_check_f(
    metrics: Metrics,
    *args: P.args,
    **kwargs: P.kwargs,
  ) -> Val:
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    f = lu.wrap_init(fun)
    f, out_tree_thunk = api_util.flatten_fun(f, in_tree)
    f, subtrace_thunk = numerics_check_subtrace(f)
    f = numerics_check_trace(f, subtrace_thunk, metrics)
    out_flat = f.call_wrapped(*args_flat)
    return tree_util.tree_unflatten(out_tree_thunk(), out_flat)

  def numerics_check_metrics_f(
    *args: P.args,
    **kwargs: P.kwargs,
  ) -> MetricsSource:
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    f = lu.wrap_init(fun)
    f, _ = api_util.flatten_fun(f, in_tree)
    f, subtrace_thunk = numerics_check_subtrace(f)
    f = numerics_check_trace(f, subtrace_thunk, {})
    pe.trace_to_jaxpr_dynamic(f, tuple(core.get_aval(x) for x in args_flat))
    return subtrace_thunk()["trace"].metric_sources

  return numerics_check_f, numerics_check_metrics_f


def metric_sources_to_metrics(metric_sources: MetricsSource) -> Metrics:
  return {
    source: (
      tuple(NumericsCheckTrace.make_metric() for _ in range(in_metrics)),
      NumericsCheckTrace.make_metric(),
    )
    for source, in_metrics in metric_sources.items()
  }
