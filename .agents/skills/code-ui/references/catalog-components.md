# UI Catalog Component Reference

> **Auto-generated** — run `bash .agents/skills/code-ui/scripts/generate-catalog-docs.sh` to refresh.
> Source: `@kumo-ai/kumo-ui-catalog` — import everything from this single entry point.
> Source directory: `packages/kumo-ui-catalog` in this repository.

## Import Pattern

```tsx
import { Button, Card, DataTable, cn } from "@kumo-ai/kumo-ui-catalog";
```

**Version:** 2.1.0 | **Explicit named exports:** 229 components/functions, 80 types

## Components by Category

### UI Primitives
`Accordion`,  `AccordionContent`,  `AccordionItem`,  `AccordionTrigger`,  `Alert`,  `AlertCtaList`,  `AlertDescription`,  `AlertTitle`,  `Badge`,  `BaseLineMetricsTable`,  `Breadcrumb`,  `BreadcrumbEllipsis`,  `BreadcrumbItem`,  `BreadcrumbLink`,  `BreadcrumbList`,  `BreadcrumbPage`,  `BreadcrumbSeparator`,  `BrowserAlert`,  `Button`,  `Calendar`,  `Card`,  `CardContent`,  `CardHeader`,  `Checkbox`,  `CodeBlock`,  `Collapsible`,  `CollapsibleContent`,  `CollapsibleTrigger`,  `ConfirmationDialog`,  `ConfusionMetricTable`,  `ConnectorChip`,  `CopyToClipBoard`,  `CustomAlert`,  `DatePicker`,  `DateTimeInput`,  `Description`,  `Dialog`,  `DialogClose`,  `DialogContent`,  `DialogDescription`,  `DialogFooter`,  `DialogHeader`,  `DialogTitle`,  `DocumentationLink`,  `DropdownMenu`,  `DropdownMenuCheckboxItem`,  `DropdownMenuContent`,  `DropdownMenuGroup`,  `DropdownMenuItem`,  `DropdownMenuLabel`,  `DropdownMenuPortal`,  `DropdownMenuRadioGroup`,  `DropdownMenuRadioItem`,  `DropdownMenuSeparator`,  `DropdownMenuShortcut`,  `DropdownMenuSub`,  `DropdownMenuSubContent`,  `DropdownMenuSubTrigger`,  `DropdownMenuTrigger`,  `EmptyEntity`,  `EmptyTable`,  `Form`,  `FormActions`,  `FormControl`,  `FormDescription`,  `FormError`,  `FormField`,  `FormSection`,  `FormWrapper`,  `HoverCard`,  `HoverCardContent`,  `HoverCardTrigger`,  `Image`,  `Input`,  `KeyPair`,  `Label`,  `ListView`,  `Markdown`,  `MetricCard`,  `OutputCard`,  `OverviewDropdown`,  `PageHeader`,  `PageNotFound`,  `PathBreadcrumb`,  `Popover`,  `PopoverAnchor`,  `PopoverButton`,  `PopoverContent`,  `PopoverTrigger`,  `ProgressInfo`,  `PropertyCard`,  `QueryCard`,  `RadioGroup`,  `RadioGroupItem`,  `RelatedJobCard`,  `ScrollArea`,  `ScrollBar`,  `SearchBar`,  `SectionHeader`,  `Select`,  `Sheet`,  `SheetClose`,  `SheetContent`,  `SheetDescription`,  `SheetFooter`,  `SheetHeader`,  `SheetTitle`,  `SheetTrigger`,  `SidebarInset`,  `SidebarMenuItem`,  `SidebarMobileTrigger`,  `SidebarProvider`,  `SidebarTrigger`,  `Skeleton`,  `Slider`,  `StartJobPopover`,  `StatsCard`,  `StatusCard`,  `StatusIcon`,  `Switch`,  `Symbol`,  `TabSwitcher`,  `Tabs`,  `TabsContent`,  `TabsList`,  `TabsTrigger`,  `Tag`,  `TagInput`,  `TagList`,  `TextArea`,  `TextInput`,  `TextLink`,  `TextScrollArea`,  `ThresholdMetricTable`,  `Title`,  `Toaster`,  `Toggle`,  `ToggleGroup`,  `ToggleGroupItem`,  `Tooltip`,  `TooltipContent`,  `TooltipProvider`,  `TooltipTrigger`,  `TruncatedTooltip`,  `Upload`,  `UploadContent`,  `UploadContext`,  `UploadDescription`,  `UploadInputElement`,  `UploadItem`,  `UploadRoot`,  `UploadTrigger`,  `ValidationStatus`,  `buttonVariants`,  `createFileUploadItems`,  `formatFileSize`,  `sidebarMenuButtonVariants`,  `useUploadContext`

### Charts (Block)
`AreaChart`,  `BarChart`,  `ChartLabel`,  `DistributionChart`,  `DivergingBarChart`,  `HeatMapChart`,  `HistogramChart`,  `PerformanceChart`,  `ScalarChart`,  `ScatterPointLineChart`

### Data Table (Block)
`DataTable`

### Layout (Block)
`GridCol`,  `GridRow`

### Sidebar (Block)
`AppSidebar`

### Graph Visualization (Block)
`HOP_COLORS`,  `HopGraph`,  `NodeGraph`,  `TableGraph`,  `TableGraphEdge`,  `TableGraphNode`,  `XaiSubgraph`

### Loading & Animation
`Bouncingball`,  `ProgressBar`,  `ProgressSpinner`,  `Spinner`

### Toast
`toast`

### Utility
`NODE_TYPE`,  `PREDICTED_NODE_TYPE_INFO`,  `PREDICTED_TYPE_LEGEND_ORDER`,  `TABLE_COLORS`,  `XAI_SUBGRAPH_LAYOUT_NODESEP`,  `buildTableColorMap`,  `clearSessionStorage`,  `cn`,  `getFromLocalStorage`,  `getFromSessionStorage`,  `getNodeFootprint`,  `getShapeForNodeType`,  `removeFromLocalStorage`,  `storeToLocalStorage`,  `storeToSessionStorage`,  `xaiSubgraphApplyLayout`

### React Flow (re export)
`BaseEdge`,  `Handle`,  `Position`,  `ReactFlowProvider`,  `SmartBezierEdge`,  `addEdge`,  `applyEdgeChanges`,  `applyNodeChanges`,  `getBezierPath`,  `getSimpleBezierPath`,  `getStraightPath`,  `useReactFlow`,  `useStore`

### Other
`DropdownCard`,  `ErrorFallback`,  `PropertiesOverviewCard`,  `TabCard`,  `ValidationCard`,  `VisuallyHidden`,  `WarningsPopover`,  `useCustomKeyPress`,  `useDebounce`,  `useIsBrowserSupported`,  `useIsMounted`,  `useLocalStorage`,  `usePrevious`,  `useWindowDimensions`,  `useWindowFocus`,  `useWithDebounce`

## Type Exports

**TanStack Table:** `CellContext`,  `ColumnDef`,  `Row`,  `RowSelectionState`,  `SortingState`,  `Table`,  `VisibilityState`

**React Flow:** `Connection`,  `Edge`,  `EdgeChange`,  `EdgeProps`,  `EdgeTypes`,  `Node`,  `NodeChange`,  `NodeProps`,  `NodeTypes`,  `OnConnect`,  `OnEdgesChange`,  `OnNodesChange`,  `ReactFlowState`

**Custom:** `AlertCtaItem`,  `AlertCtaLevel`,  `AlertCtaListProps`,  `ChipNodeData`,  `ConfirmButtonVariant`,  `ConfirmationDialogProps`,  `CustomAlertProps`,  `DialogRef`,  `DocumentationLinkProps`,  `DropdownCardProps`,  `EmptyEntityData`,  `ErrorFallbackProps`,  `FileUploadItem`,  `GridColProps`,  `GridRowProps`,  `HeatMapData`,  `HeatMapLineData`,  `HopGraphEdge`,  `HopGraphNode`,  `HopGraphProps`,  `JobOverviewProps`,  `MetricItem`,  `NodeShapeVariant`,  `NodeStatus`,  `OutputCardContent`,  `OverviewProperties`,  `PageHeaderProps`,  `PathBreadcrumbProps`,  `PredictedNodeType`,  `PredictionItem`,  `PredictionItemCol`,  `ProgressInfoProps`,  `ScalarChartData`,  `ScatterLineData`,  `ScatterPointData`,  `SelectOption`,  `SideBarData`,  `StatusIconProps`,  `SubgraphNodeData`,  `SubgraphNodeInfo`,  `TableGraphNodeData`,  `TagInputProps`,  `TagListProps`,  `TagProps`,  `TargetHistogram`,  `UploadContentProps`,  `UploadContextType`,  `UploadDescriptionProps`,  `UploadInputElementProps`,  `UploadItemProps`,  `UploadProps`,  `UploadRootProps`,  `UploadStatus`,  `UploadTriggerProps`,  `ValidationCardProps`,  `ValidationState`,  `ValidationStatusProps`,  `WarningsPopoverProps`,  `XaiSubgraphPredictions`,  `XaiSubgraphProps`

---

> **Note:** This file lists explicit named exports from the entry point; wildcard
> re-exports are not expanded. For detailed props documentation, see the
> [Storybook](http://localhost:6006) or read component source files in the in-tree catalog.
> Keep authored props guidance in ordinary documentation; regeneration replaces
> this entire index.
