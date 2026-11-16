// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/ui/markdown";
import { StatusCard } from "@/components/ui/status-card";
import {
    AutoRenew,
    BrightnessAlert,
    CheckCircle,
    NextPlan,
    TabClose
} from "@/icons";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof StatusCard> = {
    title: "UI/StatusCard",
    component: StatusCard,
    tags: ["autodocs"],
    argTypes: {
        variant: {
            control: "select",
            options: ["queued", "done", "failed", "running", "cancelled"]
        },
        className: {
            control: { type: "text" },
            defaultValue: ""
        }
    }
} satisfies Meta<typeof StatusCard>;

export default meta;
type Story = StoryObj<typeof StatusCard>;

export const QueuedStatusCard: Story = {
    args: {
        variant: "queued",
        title: "This job is queued.",
        titleIcon: <NextPlan className="fill-yellow-700" />,
        enableDivider: true,
        contentChild: (
            <>
                <Button>{"Cancel this job"}</Button>
            </>
        )
    }
};

export const RunningStatusCard: Story = {
    args: {
        variant: "running",
        title: "Generating Training data.",
        titleIcon: <AutoRenew className="fill-indigo-600" />,
        enableDivider: true,
        contentChild: (
            <>
                <Button>{"Cancel this job"}</Button>
            </>
        )
    }
};

export const DoneStatusCard: Story = {
    args: {
        variant: "done",
        title: "Training table Generated",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <div className="text-sm text-green-700">
                The model has been trained successfully.
            </div>
        )
    }
};

export const CancelledStatusCard: Story = {
    args: {
        variant: "cancelled",
        title: "This job was cancelled by the user.",
        titleIcon: <TabClose className="fill-gray-800" />,
        contentChild: (
            <div className="text-sm text-gray-800">
                The job was manually cancelled by the user.
            </div>
        )
    }
};

export const FailureStatusCard: Story = {
    args: {
        variant: "failed",
        title: "This job failed with following errors",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <div className="text-sm text-red-700">
                Graph edges have low match rate. Duplicate primary keys found.
            </div>
        )
    }
};

const overflowTitle =
    "LENGTHY TITLE WITH  S    P    A    C    E     S and Newlines: Training table generation with Job identifier 6397b Failed, as a dependency job TRAINING TABLE GENERATION with ID (6397b) encountered an error.\nTRAINING TABLE GENERATION failed.\n";
const overflowContent =
    "LENGTHY CONTENT WITH  S    P    A    C    E     S and Newlines: This is a lengthy error message .    Kumo encountered following errors while ingesting table:  'hm_transactions' . \n (1) Error in splitting the training data into valid split sets. \n (2) Not enough timestamps (1) for 3 splits. Each split must contain at least one  timestamp. \n (3) Message 3. \n (4) Message 4. \n (5) Message 5. \n (6) Message 6. \n (7) Message 7. \n (8) Message 8.";
export const FailureStatusCardOverflow: Story = {
    args: {
        variant: "failed",
        title: overflowTitle,
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <div className="text-sm text-red-700">{overflowContent}</div>
        )
    }
};

// Markdown Examples

export const DoneWithMarkdownBasic: Story = {
    args: {
        variant: "done",
        title: "Training Completed Successfully",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <Markdown size="sm" className="text-green-700 [&_*]:text-green-700">
                {`Your model has been **successfully trained** with the following results:

- Accuracy: *95.3%*
- Training time: *12 minutes*
- Dataset size: *10,000 rows*

You can now start making predictions.`}
            </Markdown>
        )
    }
};

export const FailureWithMarkdownCodeBlock: Story = {
    args: {
        variant: "failed",
        title: "Job Failed - Error Details",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-red-700 [&_*]:text-red-700 [&_code]:bg-red-50 [&_pre]:bg-red-50">
                {`The job encountered an error during execution:

**Error Code:** \`VALIDATION_ERROR_001\`

**Stack Trace:**
\`\`\`
ValueError: Invalid column type detected
  at validateSchema (validator.py:45)
  at processData (processor.py:123)
\`\`\`

**Possible Solutions:**
- Check your data types
- Ensure all required columns are present
- Review the validation rules`}
            </Markdown>
        )
    }
};

export const RunningWithMarkdownSteps: Story = {
    args: {
        variant: "running",
        title: "Training in Progress",
        titleIcon: <AutoRenew className="fill-indigo-600" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-indigo-600 [&_*]:text-indigo-600 [&_code]:bg-indigo-50 [&_strong]:font-semibold">
                {`### Current Progress

**Status:** Processing batch 3 of 10

**Completed Steps:**
1. Data validation ✓
2. Feature engineering ✓
3. Model initialization ✓

**In Progress:**
- Training model with hyperparameters

**Pending:**
- Model evaluation
- Results compilation
- Generating reports

Estimated time remaining: ~15 minutes`}
            </Markdown>
        )
    }
};

export const QueuedWithMarkdownInfo: Story = {
    args: {
        variant: "queued",
        title: "Job Queued - Position #3",
        titleIcon: <NextPlan className="fill-yellow-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-yellow-700 [&_*]:text-yellow-700 [&_code]:bg-yellow-50">
                {`Your job has been queued and will start soon.

**Job Details:**
- Job ID: \`job-1234-5678\`
- Priority: Normal
- Estimated start time: ~5 minutes

**What happens next?**
1. System allocates resources
2. Job begins processing
3. You'll receive a notification when complete

> **Tip:** You can monitor the queue status in the Jobs dashboard.`}
            </Markdown>
        )
    }
};

export const FailureWithMarkdownTable: Story = {
    args: {
        variant: "failed",
        title: "Validation Failed - Multiple Issues",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-red-700 [&_*]:text-red-700 [&_table]:border-red-300 [&_td]:border-red-300 [&_th]:border-red-300 [&_th]:bg-red-50">
                {`The following validation errors were found:

| Column | Issue | Count |
|--------|-------|-------|
| \`user_id\` | Duplicate values | 45 |
| \`email\` | Invalid format | 12 |
| \`age\` | Out of range | 8 |
| \`created_at\` | Missing values | 23 |

**Total Errors:** 88

**Action Required:** Please fix these issues and re-upload your data.`}
            </Markdown>
        )
    }
};

export const DoneWithMarkdownDocumentation: Story = {
    args: {
        variant: "done",
        title: "Deployment Complete - Next Steps",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-green-700 [&_*]:text-green-700 [&_a]:text-green-900 [&_a]:underline [&_code]:bg-green-50">
                {`### Deployment Successful! 🎉

Your model is now live and ready to receive requests.

**API Endpoint:**
\`\`\`
https://api.example.com/v1/predict
\`\`\`

**Authentication:**
Use your API key in the header:
\`\`\`bash
Authorization: Bearer YOUR_API_KEY
\`\`\`

**Example Request:**
\`\`\`json
{
  "features": [1.2, 3.4, 5.6],
  "model_version": "v1.0"
}
\`\`\`

**Monitoring:**
- View real-time metrics in the dashboard
- Set up alerts for anomalies
- Check logs for debugging

For more information, visit our [documentation](https://docs.example.com).`}
            </Markdown>
        )
    }
};

// Additional Use Cases

export const DoneWithTaskList: Story = {
    args: {
        variant: "done",
        title: "Pre-deployment Checklist Complete",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <Markdown size="sm" className="text-green-700 [&_*]:text-green-700">
                {`All pre-deployment checks passed successfully:

- [x] Code review approved
- [x] All tests passing
- [x] Security scan complete
- [x] Performance benchmarks met
- [x] Documentation updated
- [x] Database migrations tested

Your application is ready for production deployment.`}
            </Markdown>
        )
    }
};

export const RunningDataProcessing: Story = {
    args: {
        variant: "running",
        title: "Processing Large Dataset",
        titleIcon: <AutoRenew className="fill-indigo-600" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-indigo-600 [&_*]:text-indigo-600 [&_code]:bg-indigo-50">
                {`Processing 1.2M records across multiple tables.

**Progress:**
- Records processed: \`850,000 / 1,200,000\`
- Success rate: \`99.8%\`
- Failed records: \`1,700\` (will be retried)

**Performance:**
- Average speed: \`5,200 records/sec\`
- Current speed: \`4,800 records/sec\`
- ETA: ~2 minutes

The system will automatically retry failed records after completion.`}
            </Markdown>
        )
    }
};

export const FailedWithSyntaxHighlighting: Story = {
    args: {
        variant: "failed",
        title: "Deployment Failed - Configuration Error",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <Markdown
                size="sm"
                enableSyntaxHighlighting
                className="text-red-700 [&_*]:text-red-700">
                {`Deployment failed due to invalid configuration in \`config.yaml\`:

\`\`\`yaml
database:
  host: localhost
  port: 5432
  # Error: missing required field 'database'
  username: admin
  password: secret123
\`\`\`

**Error:** Missing required configuration field \`database.database\`

**Fix:** Add the database name to your configuration:

\`\`\`yaml
database:
  host: localhost
  port: 5432
  database: production_db  # Add this line
  username: admin
  password: secret123
\`\`\``}
            </Markdown>
        )
    }
};

export const QueuedWithPriority: Story = {
    args: {
        variant: "queued",
        title: "High Priority Job Queued",
        titleIcon: <NextPlan className="fill-yellow-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-yellow-700 [&_*]:text-yellow-700 [&_code]:bg-yellow-50">
                {`Your high-priority job has been added to the queue.

| Metric | Value |
|--------|-------|
| Position | #2 |
| Priority | High |
| Wait Time | ~2 min |
| Resource Req. | 4 CPU, 8GB RAM |

**Jobs ahead in queue:**
1. Job \`#8821\` - Model training (98% complete)

Once resources are available, your job will start automatically.`}
            </Markdown>
        )
    }
};

export const CancelledBySystem: Story = {
    args: {
        variant: "cancelled",
        title: "Job Cancelled - System Timeout",
        titleIcon: <TabClose className="fill-gray-800" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-gray-800 [&_*]:text-gray-800 [&_code]:bg-gray-50">
                {`This job was automatically cancelled after exceeding the maximum execution time.

**Job Information:**
- Job ID: \`job-9876-5432\`
- Started: 2 hours ago
- Max runtime: 2 hours
- Cancellation reason: Timeout

**Next Steps:**
- Review your job configuration
- Consider breaking the job into smaller tasks
- Increase timeout limit if needed
- Check logs for bottlenecks`}
            </Markdown>
        )
    }
};

export const DoneWithMetrics: Story = {
    args: {
        variant: "done",
        title: "Model Training Complete with Metrics",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-green-700 [&_*]:text-green-700 [&_code]:bg-green-50">
                {`Training completed successfully with excellent results!

### Performance Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 97.2% | 95.8% | 96.1% |
| Precision | 96.5% | 94.9% | 95.3% |
| Recall | 97.8% | 96.2% | 96.7% |
| F1 Score | 97.1% | 95.5% | 96.0% |

### Training Details
- Total epochs: \`50\`
- Best epoch: \`42\`
- Training time: \`1h 23m\`
- Dataset size: \`50,000 samples\`

Model saved to: \`models/best_model_v2.pkl\``}
            </Markdown>
        )
    }
};

export const FailedDependencyError: Story = {
    args: {
        variant: "failed",
        title: "Job Failed - Dependency Error",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <Markdown
                size="sm"
                enableSyntaxHighlighting
                className="text-red-700 [&_*]:text-red-700">
                {`Job failed due to missing dependencies. Cannot proceed with execution.

**Failed Dependencies:**

\`\`\`bash
❌ Job: data-preprocessing (job-1234)
   Status: Failed
   Error: Missing input file

❌ Job: feature-extraction (job-1235)
   Status: Blocked
   Waiting for: job-1234

❌ Job: model-training (job-1236)
   Status: Blocked
   Waiting for: job-1235
\`\`\`

**Resolution:**
1. Check if \`data-preprocessing\` job completed
2. Verify input files exist in: \`/data/raw/\`
3. Re-run failed jobs in order
4. Monitor dependency chain`}
            </Markdown>
        )
    }
};

export const RunningWithResourceUsage: Story = {
    args: {
        variant: "running",
        title: "Training Neural Network",
        titleIcon: <AutoRenew className="fill-indigo-600" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-indigo-600 [&_*]:text-indigo-600 [&_code]:bg-indigo-50">
                {`Deep learning model training in progress...

**Current Epoch:** 23 / 100

**Metrics:**
- Loss: \`0.1247\` (↓ improving)
- Accuracy: \`89.3%\` (↑ improving)
- Learning rate: \`0.0001\`

**Resource Usage:**
- GPU: \`87%\` (RTX 4090)
- Memory: \`18.2 GB / 24 GB\`
- CPU: \`45%\` (16 cores)
- Disk I/O: \`234 MB/s\`

**Time:**
- Elapsed: 1h 12m
- Remaining: ~3h 45m
- Epoch time: ~3.5 min/epoch`}
            </Markdown>
        )
    }
};

export const QueuedScheduled: Story = {
    args: {
        variant: "queued",
        title: "Scheduled Job - Pending Start",
        titleIcon: <NextPlan className="fill-yellow-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-yellow-700 [&_*]:text-yellow-700 [&_code]:bg-yellow-50">
                {`This job is scheduled to run at a specific time.

**Schedule Information:**
- Scheduled start: \`2025-12-09 22:00 UTC\`
- Current time: \`2025-12-09 14:30 UTC\`
- Time until start: \`7 hours 30 minutes\`

**Job Configuration:**
- Type: Batch processing
- Recurrence: Daily
- Priority: Normal
- Estimated duration: ~45 minutes

The job will start automatically at the scheduled time. You can cancel or reschedule anytime before execution.`}
            </Markdown>
        )
    }
};

export const FailedOutOfMemory: Story = {
    args: {
        variant: "failed",
        title: "Job Failed - Out of Memory",
        titleIcon: <BrightnessAlert className="fill-red-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-red-700 [&_*]:text-red-700 [&_code]:bg-red-50">
                {`Job terminated due to insufficient memory.

**Error Details:**
- Error type: \`OutOfMemoryError\`
- Requested: \`32 GB\`
- Available: \`16 GB\`
- Location: \`model_training.py:line 156\`

**Memory Usage at Failure:**
- Batch size: \`512\`
- Model parameters: \`2.4B\`
- Activations: \`~18 GB\`
- Total required: \`~32 GB\`

**Recommended Fixes:**
- ~~Use larger instance~~ Reduce batch size to \`256\` or \`128\`
- Enable gradient checkpointing
- Use mixed precision training (FP16)
- Reduce model size or use model parallelism
- Clear cache between batches`}
            </Markdown>
        )
    }
};

export const DoneWithNextActions: Story = {
    args: {
        variant: "done",
        title: "Data Migration Complete",
        titleIcon: <CheckCircle className="fill-green-700" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-green-700 [&_*]:text-green-700 [&_code]:bg-green-50">
                {`Successfully migrated all data to the new database.

**Migration Summary:**
- Tables migrated: \`47\`
- Records transferred: \`12,487,392\`
- Duration: \`2h 15m\`
- Data integrity: ✅ Verified

**What Changed:**
- New schema version: \`v3.2.0\`
- Indexes created: \`128\`
- Foreign keys updated: \`64\`

**Next Actions:**
- [ ] Update application configuration
- [ ] Run smoke tests on new database
- [ ] Monitor query performance
- [ ] Archive old database (after 7 days)
- [ ] Update team documentation

Old database will remain available for rollback until Dec 16, 2025.`}
            </Markdown>
        )
    }
};

export const CancelledByUser: Story = {
    args: {
        variant: "cancelled",
        title: "Job Cancelled by User",
        titleIcon: <TabClose className="fill-gray-800" />,
        contentChild: (
            <Markdown
                size="sm"
                className="text-gray-800 [&_*]:text-gray-800 [&_code]:bg-gray-50">
                {`This job was manually cancelled by **john.doe@example.com**.

**Job Progress Before Cancellation:**
- Completed: 67%
- Records processed: \`67,000 / 100,000\`
- Time elapsed: 15 minutes

**Partial Results:**
- Successfully processed records: \`65,450\`
- Failed records: \`1,550\`
- Partial output saved to: \`/results/partial/job_1234\`

You can resume this job from the last checkpoint or start a new job with updated parameters.`}
            </Markdown>
        )
    }
};

export const MinimalWithoutContent: Story = {
    args: {
        variant: "running",
        title: "Processing...",
        titleIcon: <AutoRenew className="fill-indigo-600" />
    }
};

export const MinimalQueued: Story = {
    args: {
        variant: "queued",
        title: "Waiting for resources",
        titleIcon: <NextPlan className="fill-yellow-700" />
    }
};

export const WithoutIcon: Story = {
    args: {
        variant: "done",
        title: "Operation completed successfully",
        contentChild: (
            <div className="text-sm text-green-700">
                All tasks finished without errors.
            </div>
        )
    }
};
