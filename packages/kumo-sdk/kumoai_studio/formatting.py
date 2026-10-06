# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.jobs import ErrorDetails


def pretty_print_error_details(error_details: ErrorDetails) -> str:
    """Pretty prints the ErrorDetails combining all the individual items.
    If there are CTAs, they are also displayed after creating
    corresponding hyperlinks.

    Arguments:
    error_details (ErrorDetails): Standard ErrorDetails response from
        get_errors APIs.
    """
    out = ""
    ctr = None
    if len(error_details.items) != 1:
        out += "Encountered multiple errors:\n"
        ctr = 1
    for error_detail in error_details.items:
        if ctr is not None:
            out += f'{ctr}.'
            ctr += 1
        if error_detail.title is not None:
            out += f'{error_detail.title}: '
        out += error_detail.description
        if error_detail.cta is not None:
            out += 'Follow the link for potential resolution:'
            f' {error_detail.cta.url}'
        out += '\n'

    return out
