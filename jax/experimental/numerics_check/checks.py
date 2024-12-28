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

from functools import partial

from jax._src import core, pjit, typing
from jax.experimental.numerics_check.numerics_check import (
  Val,
  register_numerics_check,
  NumericsCheckTrace,
  numerics_check_subtrace,
)
from jax._src.interpreters import partial_eval as pe
from jax._src import linear_util as lu


def _numerics_check_jaxpr_trace(
  trace: NumericsCheckTrace,
  jaxpr: core.ClosedJaxpr,
) -> core.ClosedJaxpr:
  f = lu.wrap_init(partial(core.eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))
  f, subtrace_thunk = numerics_check_subtrace(f, core.TraceTag(), trace.metrics)
  jaxpr_, _, consts, () = pe.trace_to_jaxpr_dynamic(f, jaxpr.in_avals)
  trace.metric_sources.update(subtrace_thunk()["trace"].metric_sources)
  return core.ClosedJaxpr(jaxpr_, consts)


@register_numerics_check(pjit.pjit_p)
def _pjit_numerics_check(
  trace: NumericsCheckTrace,
  in_metrics: tuple[typing.Array, ...],
  out_metric: typing.Array,
  *args: Val,
  jaxpr: core.ClosedJaxpr,
  **kwargs: Val,
) -> Val:
  del in_metrics, out_metric
  jaxpr = _numerics_check_jaxpr_trace(trace, jaxpr)
  out_vals = pjit.pjit_p.bind(*args, jaxpr=jaxpr, **kwargs)
  return out_vals
