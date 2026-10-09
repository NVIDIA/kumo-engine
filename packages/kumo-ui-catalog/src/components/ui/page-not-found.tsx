// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SectionHeader } from "./section-header";
import { TextLink } from "./text-link";

interface PageNotFoundProps {
    title?: string;
    redirectLink?: string;
}

/**
 * Customized component for not implemented/invalid pages
 *
 * @param title Title of the page
 * @param redirectLink Path the "here" link navigates to
 * @returns ```<PageNotFound></PageNotFound>```
 */
function PageNotFound({
    title = "404 - Page Not Found",
    redirectLink = "/"
}: PageNotFoundProps) {
    return (
        <div
            data-slot="page-not-found"
            className="flex h-full w-full flex-row items-center justify-center">
            <div className="flex flex-col items-center gap-4">
                <SectionHeader
                    title={title}
                    titleClassName="text-center"
                    titleSize="xxl"
                />
                <SectionHeader
                    title={"The page you are looking for does not exist."}
                    titleSize="xl"
                    titleClassName="text-center !font-medium"
                />
                <div className="clickable text-title-primary text-center text-xl leading-normal">
                    <span>
                        Click{" "}
                        <TextLink
                            id="home_page"
                            pageLink={redirectLink}
                            title="here"
                        />{" "}
                        to go back
                    </span>
                </div>
            </div>
        </div>
    );
}

export { PageNotFound };
