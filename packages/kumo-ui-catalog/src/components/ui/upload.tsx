// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { cva } from "class-variance-authority";

import {
    CheckCircle,
    Close,
    Error as ErrorIcon,
    FilePresent,
    Replay
} from "@/icons";
import { cn } from "@/lib/utils";

import { Button } from "./button";

export type UploadStatus = "uploading" | "success" | "error";

export interface FileUploadItem {
    /** An ID we generate for the file. */
    id: string;
    /** The raw `File` object selected by the user. */
    file: File;
    /** An optional error message to display when status is "error". */
    errorMessage?: string;
    /** The status of the file upload. */
    status: UploadStatus;
    /** The amount of the file that has been uploaded, in bytes. */
    uploadedBytes?: number;
    /** Whether to hide the preview of the file. */
    hidePreview?: boolean;
}

type UploadValueProps =
    | {
          multiple?: false;
          defaultValue?: FileUploadItem;
          /**
           * The selected file. Pass `null` to stay controlled with nothing
           * selected: `undefined` means "this upload manages its own state",
           * so a controlled upload that used it to clear the selection would
           * hand control back and resurface the file it had just removed.
           */
          value?: FileUploadItem | null;
          /**
           * Receives `undefined` once the last file is removed.
           */
          onValueChange?: (file: FileUploadItem | undefined) => void;
      }
    | {
          multiple: true;
          defaultValue?: FileUploadItem[];
          /** Pass `[]` to stay controlled with nothing selected. */
          value?: FileUploadItem[] | null;
          onValueChange?: (files: FileUploadItem[]) => void;
      };

const toArray = (
    value: FileUploadItem | FileUploadItem[] | null | undefined
): FileUploadItem[] => (!value ? [] : Array.isArray(value) ? value : [value]);

/** Formats a byte count the way the KUI upload item does — "1.2 MB". */
// eslint-disable-next-line react-refresh/only-export-components
export const formatFileSize = (bytes: number): string => {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "0 B";
    }
    const units = ["B", "KB", "MB", "GB", "TB"];
    const exponent = Math.min(
        Math.floor(Math.log(bytes) / Math.log(1000)),
        units.length - 1
    );
    const size = bytes / Math.pow(1000, exponent);
    return `${exponent === 0 ? size : Number(size.toFixed(1))} ${units[exponent]}`;
};

let fileIdCounter = 0;
const nextFileId = () => `upload-file-${++fileIdCounter}`;

/** Wraps raw `File` objects as upload items in the "uploading" status. */
// eslint-disable-next-line react-refresh/only-export-components
export const createFileUploadItems = (files: File[]): FileUploadItem[] =>
    files.map((file) => ({
        id: nextFileId(),
        file,
        status: "uploading" as const,
        uploadedBytes: 0
    }));

export interface UploadContextType {
    disabled: boolean;
    files: FileUploadItem[];
    multiple: boolean;
    addFiles: (files: File[]) => void;
    removeFile: (file: FileUploadItem) => void;
    retryFile?: (file: FileUploadItem) => void;
}

const UploadContext = React.createContext<UploadContextType | null>(null);

// eslint-disable-next-line react-refresh/only-export-components
export const useUploadContext = (): UploadContextType => {
    const context = React.useContext(UploadContext);
    if (!context) {
        throw new Error("Upload components must be used within an UploadRoot");
    }
    return context;
};

export type UploadRootProps = {
    children: React.ReactNode;
    /**
     * Disables the upload, preventing file selection, drag-and-drop, and item
     * actions.
     * @defaultValue false
     */
    disabled?: boolean;
    /**
     * Called when the retry button on a failed file is activated. Provide this
     * handler to render the retry affordance.
     */
    onFileRetry?: (file: FileUploadItem) => void;
    /**
     * Called when a file is removed. The file is removed from internal state
     * regardless of whether this handler is provided.
     */
    onFileRemove?: (file: FileUploadItem) => void;
} & UploadValueProps;

/**
 * The state container of a composed upload. Tracks the selected files and
 * shares them with the trigger and content.
 */
function UploadRoot({
    children,
    defaultValue,
    disabled = false,
    multiple = false,
    onFileRemove,
    onFileRetry,
    onValueChange,
    value
}: UploadRootProps) {
    const [uncontrolled, setUncontrolled] = React.useState<FileUploadItem[]>(
        () => toArray(defaultValue)
    );
    const isControlled = value !== undefined;
    const files = isControlled ? toArray(value) : uncontrolled;

    const commit = React.useCallback(
        (next: FileUploadItem[]) => {
            if (!isControlled) {
                setUncontrolled(next);
            }
            if (multiple) {
                (onValueChange as (files: FileUploadItem[]) => void)?.(next);
            } else {
                (onValueChange as (file: FileUploadItem) => void)?.(next[0]);
            }
        },
        [isControlled, multiple, onValueChange]
    );

    const addFiles = React.useCallback(
        (incoming: File[]) => {
            if (disabled || incoming.length === 0) {
                return;
            }
            const added = createFileUploadItems(
                multiple ? incoming : incoming.slice(0, 1)
            );
            commit(multiple ? [...files, ...added] : added);
        },
        [commit, disabled, files, multiple]
    );

    const removeFile = React.useCallback(
        (target: FileUploadItem) => {
            if (disabled) {
                return;
            }
            commit(files.filter((item) => item.id !== target.id));
            onFileRemove?.(target);
        },
        [commit, disabled, files, onFileRemove]
    );

    const retryFile = React.useMemo(
        () =>
            onFileRetry
                ? (target: FileUploadItem) => {
                      if (!disabled) {
                          onFileRetry(target);
                      }
                  }
                : undefined,
        [disabled, onFileRetry]
    );

    const context = React.useMemo<UploadContextType>(
        () => ({ disabled, files, multiple, addFiles, removeFile, retryFile }),
        [addFiles, disabled, files, multiple, removeFile, retryFile]
    );

    return (
        <UploadContext.Provider value={context}>
            {children}
        </UploadContext.Provider>
    );
}

const uploadTriggerVariants = cva(
    // KUI .nv-input-shell + .nv-upload-trigger, light theme. Resting border is
    // --border-color-interaction-base (#0000004d); the file-input shell swaps
    // the fixed 40px height for auto height with 24px block padding.
    "text-title-primary relative flex w-full cursor-pointer flex-col items-center justify-center rounded-sm px-3 py-6 text-center text-sm font-normal leading-[1.14286] transition-colors",
    {
        variants: {
            kind: {
                flat: "border-kui-border-interaction-base bg-white border",
                floating: "border-0 bg-transparent"
            },
            status: {
                error: "border-kui-danger",
                success: "border-kui-success",
                none: ""
            },
            dragged: {
                true: "border-kui-hover-border bg-kui-bg-hover",
                false: ""
            },
            disabled: {
                true: "border-kui-border-disabled bg-kui-bg-disabled text-kui-fg-disabled cursor-not-allowed",
                false: "hover:bg-kui-bg-hover"
            }
        },
        defaultVariants: {
            kind: "flat",
            status: "none",
            dragged: false,
            disabled: false
        }
    }
);

export interface UploadTriggerProps
    extends Omit<React.ComponentPropsWithRef<"label">, "children"> {
    /**
     * Bordered shell (flat) or borderless (floating).
     *
     * Declared here rather than picked off `VariantProps`: cva's `ClassProp` is
     * a union, so `Pick<VariantProps<...>, "kind">` resolves to a required
     * property and every consumer of `UploadProps` had to pass a `kind` it
     * already had a default for.
     *
     * @defaultValue "flat"
     */
    kind?: "flat" | "floating";
    children?: React.ReactNode;
    /** Replaces the default "Choose file"/"Choose files" anchor label. */
    slotAnchor?: React.ReactNode;
    /**
     * Trailing text shown next to the anchor that completes the call-to-action.
     * @defaultValue " or drag and drop it here."
     */
    slotHeaderText?: React.ReactNode;
    /** Renders the field in a validation state. */
    status?: "error" | "success";
}

/**
 * The `<label>` element that wraps the upload input. Provides the click target
 * and the drag-and-drop drop zone.
 */
const UploadTrigger = React.forwardRef<HTMLLabelElement, UploadTriggerProps>(
    (
        {
            children,
            className,
            kind = "flat",
            slotAnchor,
            slotHeaderText = " or drag and drop it here.",
            status,
            onDragEnter,
            onDragLeave,
            onDragOver,
            onDrop,
            ...props
        },
        ref
    ) => {
        const { addFiles, disabled, multiple } = useUploadContext();
        const [dragged, setDragged] = React.useState(false);
        // Drag events fire for every nested element, so count enters and leaves
        // instead of toggling a boolean.
        const dragDepth = React.useRef(0);

        const endDrag = () => {
            dragDepth.current = 0;
            setDragged(false);
        };

        return (
            <label
                ref={ref}
                data-slot="upload-trigger"
                data-dragged-over={dragged || undefined}
                data-status={status}
                data-disabled={disabled || undefined}
                className={cn(
                    uploadTriggerVariants({
                        kind,
                        status: status ?? "none",
                        dragged: dragged && !disabled,
                        disabled
                    }),
                    className
                )}
                onDragEnter={(event) => {
                    onDragEnter?.(event);
                    if (disabled) return;
                    dragDepth.current += 1;
                    setDragged(true);
                }}
                onDragOver={(event) => {
                    onDragOver?.(event);
                    if (disabled) return;
                    // Without this the browser opens the file instead.
                    event.preventDefault();
                }}
                onDragLeave={(event) => {
                    onDragLeave?.(event);
                    if (disabled) return;
                    dragDepth.current -= 1;
                    if (dragDepth.current <= 0) {
                        endDrag();
                    }
                }}
                onDrop={(event) => {
                    onDrop?.(event);
                    if (disabled) return;
                    event.preventDefault();
                    endDrag();
                    addFiles(Array.from(event.dataTransfer.files));
                }}
                {...props}>
                <span>
                    <span
                        data-slot="upload-trigger-anchor"
                        className={cn(
                            "decoration-brand-green underline underline-offset-4",
                            disabled && "no-underline"
                        )}>
                        {slotAnchor ??
                            (multiple ? "Choose files" : "Choose file")}
                    </span>
                    {slotHeaderText}
                </span>
                {children}
            </label>
        );
    }
);
UploadTrigger.displayName = "UploadTrigger";

export interface UploadInputElementProps
    extends React.ComponentPropsWithRef<"input"> {
    /**
     * Comma-separated list of MIME types or file extensions the picker should
     * allow.
     */
    accept?: string;
}

/**
 * The hidden `<input type="file">` element that drives file selection inside an
 * upload trigger.
 */
const UploadInputElement = React.forwardRef<
    HTMLInputElement,
    UploadInputElementProps
>(({ className, onChange, ...props }, ref) => {
    const { addFiles, disabled, multiple } = useUploadContext();

    return (
        <input
            ref={ref}
            type="file"
            data-slot="upload-input-element"
            className={cn("sr-only", className)}
            disabled={disabled}
            multiple={multiple}
            onChange={(event) => {
                onChange?.(event);
                addFiles(Array.from(event.target.files ?? []));
                // Clear the input so re-picking the same file still fires.
                event.target.value = "";
            }}
            {...props}
        />
    );
});
UploadInputElement.displayName = "UploadInputElement";

export type UploadDescriptionProps = React.ComponentPropsWithRef<"div">;

/**
 * Helper text rendered inside the upload trigger, typically used for accepted
 * file types or size limits.
 */
const UploadDescription = React.forwardRef<
    HTMLDivElement,
    UploadDescriptionProps
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        data-slot="upload-description"
        className={cn(
            "inline-flex w-full flex-wrap justify-center gap-1 pt-1.5 text-[0.625rem] font-normal leading-[1.5]",
            className
        )}
        {...props}
    />
));
UploadDescription.displayName = "UploadDescription";

const uploadItemVariants = cva(
    "border-kui-border-base text-title-primary flex overflow-hidden rounded-sm border bg-white font-normal not-italic",
    {
        variants: {
            kind: {
                card: "gap-4 p-3",
                media: "relative flex h-25 w-25 items-center justify-center p-2"
            },
            status: {
                error: "border-kui-danger",
                success: "",
                uploading: ""
            }
        },
        defaultVariants: { kind: "card", status: "uploading" }
    }
);

export interface UploadItemProps
    extends Omit<React.ComponentPropsWithRef<"div">, "children"> {
    /** The file entry to render, including its `File` and upload status. */
    item: FileUploadItem;
    /**
     * The layout used to render the file.
     * - "card" — a detailed row with file name, size, and progress.
     * - "media" — a compact thumbnail tile.
     * @defaultValue "card"
     */
    kind?: "card" | "media" | null;
    /**
     * Renders a custom preview in place of the default thumbnail. Receives the
     * file and the decoded preview URL when available.
     */
    renderPreview?: (
        file: File,
        previewUrl: string | undefined
    ) => React.ReactNode;
    /**
     * Whether to render a preview thumbnail for the file.
     * @defaultValue true
     */
    showPreview?: boolean;
}

/** Object URL for image files only — everything else falls back to an icon. */
const useObjectUrl = (file: File, enabled: boolean) => {
    const [url, setUrl] = React.useState<string>();

    React.useEffect(() => {
        if (!enabled || !file.type.startsWith("image/")) {
            setUrl(undefined);
            return;
        }
        const objectUrl = URL.createObjectURL(file);
        setUrl(objectUrl);
        return () => URL.revokeObjectURL(objectUrl);
    }, [enabled, file]);

    return url;
};

/** A single file row (card) or tile (media) inside the upload list. */
const UploadItem = React.forwardRef<HTMLDivElement, UploadItemProps>(
    (
        {
            className,
            item,
            kind = "card",
            renderPreview,
            showPreview,
            ...props
        },
        ref
    ) => {
        const { disabled, removeFile, retryFile } = useUploadContext();
        const withPreview = (showPreview ?? true) && !item.hidePreview;
        const previewUrl = useObjectUrl(item.file, withPreview);
        const isMedia = kind === "media";
        const total = item.file.size;
        const uploaded = item.uploadedBytes ?? 0;

        const preview = renderPreview ? (
            renderPreview(item.file, previewUrl)
        ) : previewUrl ? (
            <img
                data-slot="upload-item-thumbnail"
                className={cn(
                    "object-cover",
                    isMedia ? "absolute inset-0 h-full w-full" : "size-8"
                )}
                src={previewUrl}
                alt=""
            />
        ) : (
            <div
                data-slot="upload-item-thumbnail"
                className={cn(
                    "border-kui-border-base text-kui-fg-base grid size-8 min-w-8 place-items-center rounded-sm border",
                    isMedia && "absolute inset-0 m-auto"
                )}>
                <FilePresent className="size-4 fill-current" />
            </div>
        );

        const statusIcon =
            item.status === "error" ? (
                <ErrorIcon
                    data-slot="upload-item-status-icon"
                    className="fill-kui-danger size-4"
                    aria-label="Upload failed"
                />
            ) : item.status === "success" ? (
                <CheckCircle
                    data-slot="upload-item-status-icon"
                    className="fill-kui-success size-4"
                    aria-label="Upload complete"
                />
            ) : (
                <span
                    data-slot="upload-item-upload-spinner"
                    role="status"
                    aria-label="Uploading"
                    className="border-kui-border-base border-t-title-primary size-3 animate-spin rounded-full border-2"
                />
            );

        const actions = (
            <div
                data-slot="upload-item-actions-group"
                className="flex w-fit flex-wrap justify-end gap-1">
                {item.status === "error" && retryFile && (
                    <Button
                        type="button"
                        variant="no-outline"
                        size="xs"
                        aria-label={`Retry ${item.file.name}`}
                        disabled={disabled}
                        onClick={() => retryFile(item)}>
                        <Replay />
                    </Button>
                )}
                <Button
                    type="button"
                    variant="no-outline"
                    size="xs"
                    aria-label={`Remove ${item.file.name}`}
                    disabled={disabled}
                    onClick={() => removeFile(item)}>
                    <Close />
                </Button>
            </div>
        );

        return (
            <div
                ref={ref}
                data-slot="upload-item"
                data-status={item.status}
                className={cn(
                    uploadItemVariants({
                        kind: kind ?? "card",
                        status: item.status
                    }),
                    className
                )}
                {...props}>
                {withPreview && preview}
                {isMedia ? (
                    <>
                        <span
                            data-slot="upload-item-top-left"
                            className="absolute left-2 top-2">
                            {statusIcon}
                        </span>
                        <span
                            data-slot="upload-item-top-right"
                            className="absolute right-2 top-2 opacity-0 transition-opacity group-hover/item:opacity-100 [.group\/item:focus-within_&]:opacity-100">
                            {actions}
                        </span>
                    </>
                ) : (
                    <div
                        data-slot="upload-item-content"
                        className="flex flex-1 flex-col justify-between gap-2">
                        <span
                            data-slot="upload-item-content-heading"
                            className="text-sm font-normal leading-[1.28571]">
                            {item.file.name}
                        </span>
                        <span
                            data-slot="upload-item-content-info"
                            className={cn(
                                "inline-flex items-center gap-1 text-[0.625rem] font-normal leading-[1.2] [&>*]:whitespace-nowrap",
                                item.status === "error"
                                    ? "text-kui-danger-subtle"
                                    : "text-kui-fg-base"
                            )}>
                            {statusIcon}
                            <span>
                                {item.status === "error"
                                    ? (item.errorMessage ?? "Upload failed")
                                    : item.status === "uploading"
                                      ? `${formatFileSize(uploaded)} of ${formatFileSize(total)}`
                                      : formatFileSize(total)}
                            </span>
                        </span>
                    </div>
                )}
                {!isMedia && actions}
            </div>
        );
    }
);
UploadItem.displayName = "UploadItem";

export interface UploadContentProps
    extends Omit<React.ComponentPropsWithRef<"div">, "children">,
        Pick<UploadItemProps, "showPreview"> {
    /**
     * The layout used to render each selected file.
     * - "card" — detailed rows showing file name, size, and progress.
     * - "media" — compact thumbnail tiles.
     * @defaultValue "card"
     */
    kind?: "card" | "media" | null;
    /** Native HTML attributes forwarded to each rendered `UploadItem`. */
    attributes?: { UploadItem?: Partial<UploadItemProps> };
}

/** The list of selected files, rendered as cards or media tiles. */
const UploadContent = React.forwardRef<HTMLDivElement, UploadContentProps>(
    ({ attributes, className, kind = "card", showPreview, ...props }, ref) => {
        const { files } = useUploadContext();

        if (files.length === 0) {
            return null;
        }

        return (
            <div
                ref={ref}
                data-slot="upload-content"
                className={cn(
                    "flex gap-2 pt-2",
                    kind === "media" ? "flex-row flex-wrap" : "flex-col",
                    className
                )}
                {...props}>
                {files.map((item) => (
                    <UploadItem
                        key={item.id}
                        item={item}
                        kind={kind}
                        showPreview={showPreview}
                        {...attributes?.UploadItem}
                        className={cn(
                            "group/item",
                            attributes?.UploadItem?.className
                        )}
                    />
                ))}
            </div>
        );
    }
);
UploadContent.displayName = "UploadContent";

export type UploadProps = Pick<
    UploadRootProps,
    "disabled" | "onFileRemove" | "onFileRetry"
> &
    UploadValueProps &
    Pick<
        UploadTriggerProps,
        "kind" | "slotAnchor" | "slotHeaderText" | "status"
    > &
    Pick<UploadInputElementProps, "accept" | "id" | "name" | "required"> &
    Pick<UploadContentProps, "showPreview"> & {
        /**
         * The layout used to display selected files.
         * @defaultValue "card"
         */
        listKind?: UploadContentProps["kind"];
        /**
         * Helper text rendered inside the trigger, such as accepted file types
         * or size limits.
         */
        children?: React.ReactNode;
        className?: string;
        /**
         * Renders the trigger inside a custom wrapper. Useful for combining the
         * input with a Tooltip or FormField.
         */
        renderInput?: (slotInput: React.ReactNode) => React.ReactNode;
        /** Native HTML attributes forwarded to the composed components. */
        attributes?: {
            UploadDescription?: UploadDescriptionProps;
            UploadInputElement?: UploadInputElementProps;
            UploadContent?: UploadContentProps;
            UploadItem?: Partial<UploadItemProps>;
        };
    };

/**
 * A file picker with drag-and-drop support that renders previews and basic
 * management controls for the selected files.
 *
 * Upload handles the UI layer only — validation beyond MIME-type filtering and
 * the actual upload request are the application's responsibility.
 */
const Upload = React.forwardRef<HTMLLabelElement, UploadProps>(
    (
        {
            accept,
            attributes,
            children,
            className,
            defaultValue,
            disabled,
            id,
            kind,
            listKind = "card",
            multiple,
            name,
            onFileRemove,
            onFileRetry,
            onValueChange,
            renderInput,
            required,
            showPreview,
            slotAnchor,
            slotHeaderText,
            status,
            value
        },
        ref
    ) => {
        const trigger = (
            <UploadTrigger
                ref={ref}
                kind={kind}
                slotAnchor={slotAnchor}
                slotHeaderText={slotHeaderText}
                status={status}
                className={className}>
                <UploadInputElement
                    accept={accept}
                    id={id}
                    name={name}
                    required={required}
                    {...attributes?.UploadInputElement}
                />
                {children && (
                    <UploadDescription {...attributes?.UploadDescription}>
                        {children}
                    </UploadDescription>
                )}
            </UploadTrigger>
        );

        return (
            <UploadRoot
                disabled={disabled}
                onFileRemove={onFileRemove}
                onFileRetry={onFileRetry}
                {...({
                    multiple,
                    defaultValue,
                    value,
                    onValueChange
                } as UploadValueProps)}>
                {renderInput ? renderInput(trigger) : trigger}
                <UploadContent
                    kind={listKind}
                    showPreview={showPreview}
                    attributes={{ UploadItem: attributes?.UploadItem }}
                    {...attributes?.UploadContent}
                />
            </UploadRoot>
        );
    }
);
Upload.displayName = "Upload";

export {
    Upload,
    UploadContent,
    UploadContext,
    UploadDescription,
    UploadInputElement,
    UploadItem,
    UploadRoot,
    UploadTrigger
};
