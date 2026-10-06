// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

interface ImageProps {
    src: string;
    /** Required alt text — exposed as `name` for backward compatibility with the kumo @ui/Image API */
    name: string;
    width?: string;
}

export const Image = ({ src, name, width = "28" }: ImageProps) => {
    return <img src={src} width={width} alt={name} />;
};
