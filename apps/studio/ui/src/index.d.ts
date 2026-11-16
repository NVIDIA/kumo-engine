// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* Have to declare module to import `.svg`, `.png` and `.jpg` files in `.tsx` file  */
declare module "*.svg" {
    const content: Element<React.SVGAttributes>;
    export default content;
}
declare module "*.jpg";
declare module "*.png";
declare module "*.gif";
