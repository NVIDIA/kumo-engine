// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Button,
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
    FormControl,
    FormField,
    FormSection,
    Label,
    Switch
} from "@kumo-ai/kumo-ui-catalog";
import { useCacheStore } from "stores/cacheStore";

const CacheConfiguration = () => {
    const { cacheConfig, setCacheConfig, resetCacheConfig } = useCacheStore();
    const [cacheConfigOpen, setCacheConfigOpen] = useState(false);
    return (
        <Collapsible open={cacheConfigOpen} onOpenChange={setCacheConfigOpen}>
            <CollapsibleTrigger asChild>
                <div className="flex justify-between items-center my-3 !my-3">
                    <h3 className="mb-0">Cache Configuration</h3>
                    <Button
                        variant="link"
                        size="sm"
                        onClick={() => setCacheConfigOpen(!cacheConfigOpen)}
                        aria-expanded={cacheConfigOpen}>
                        {cacheConfigOpen ? "Hide" : "Show"}
                    </Button>
                </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="w-full">
                <div className="w-[96%]">
                    <FormField>
                        <div className="flex items-center gap-3 mb-2 ">
                            <div className="w-full sm:w-1/4 ">
                                <Label
                                    htmlFor="enable-caching"
                                    className="mb-0">
                                    Enable Caching
                                </Label>
                            </div>
                            <div className="w-full sm:w-1/2 ">
                                <Switch
                                    id="enable-caching"
                                    variant="primary"
                                    checked={cacheConfig.enabled}
                                    onCheckedChange={(checked) => {
                                        setCacheConfig({
                                            ...cacheConfig,
                                            enabled: checked
                                        });
                                    }}
                                />
                            </div>
                            <div className="w-full sm:w-1/3 justify-end ">
                                <Button
                                    variant="light-outline"
                                    size="sm"
                                    onClick={() => resetCacheConfig()}>
                                    Reset
                                </Button>
                            </div>
                        </div>
                    </FormField>

                    {cacheConfig.enabled && (
                        <FormSection>
                            <h4 className="text-base font-semibold mb-2">
                                Cache Times (minutes)
                            </h4>

                            <FormField>
                                <div className="flex items-center justify-between w-full  ">
                                    <Label
                                        htmlFor="short-cache"
                                        className="whitespace-nowrap mb-0">
                                        Short Cache (Jobs, live data)
                                    </Label>
                                    <FormControl
                                        id="short-cache"
                                        className="w-80"
                                        type="number"
                                        min="0"
                                        value={cacheConfig.staleTimeShort}
                                        onChange={(
                                            e: React.ChangeEvent<HTMLInputElement>
                                        ) => {
                                            setCacheConfig({
                                                ...cacheConfig,
                                                staleTimeShort:
                                                    parseInt(e.target.value) ||
                                                    0
                                            });
                                        }}
                                    />
                                </div>
                            </FormField>

                            <FormField>
                                <div className="flex items-center justify-between w-full ">
                                    <Label
                                        htmlFor="medium-cache"
                                        className="whitespace-nowrap mb-0">
                                        Medium Cache (Most APIs)
                                    </Label>
                                    <FormControl
                                        className="w-80"
                                        id="medium-cache"
                                        type="number"
                                        min="0"
                                        value={cacheConfig.staleTimeMedium}
                                        onChange={(
                                            e: React.ChangeEvent<HTMLInputElement>
                                        ) => {
                                            setCacheConfig({
                                                ...cacheConfig,
                                                staleTimeMedium:
                                                    parseInt(e.target.value) ||
                                                    0
                                            });
                                        }}
                                    />
                                </div>
                            </FormField>

                            <FormField>
                                <div className="flex items-center justify-between w-full">
                                    <Label
                                        htmlFor="long-cache"
                                        className="whitespace-nowrap mb-0">
                                        Long Cache (Infrequently changing data)
                                    </Label>
                                    <FormControl
                                        className="w-80"
                                        id="long-cache"
                                        type="number"
                                        min="0"
                                        value={cacheConfig.staleTimeLong}
                                        onChange={(
                                            e: React.ChangeEvent<HTMLInputElement>
                                        ) => {
                                            setCacheConfig({
                                                ...cacheConfig,
                                                staleTimeLong:
                                                    parseInt(e.target.value) ||
                                                    0
                                            });
                                        }}
                                    />
                                </div>
                            </FormField>
                        </FormSection>
                    )}
                </div>
            </CollapsibleContent>
        </Collapsible>
    );
};

export default CacheConfiguration;
