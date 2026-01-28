import os
from dcase_util.containers import MetaDataContainer
from sed_eval.sound_event import EventBasedMetrics

from dcase_util.containers import MetaDataContainer
from collections import defaultdict

def merge_events(events, gap_threshold=0.2):
    merged = MetaDataContainer()
    grouped = defaultdict(list)

    # Group events by (filename, event_label)
    for item in events:
        key = (item['filename'], item['event_label'])
        grouped[key].append(item)

    # Merge within each group
    for (filename, label), group in grouped.items():
        group = sorted(group, key=lambda x: x.onset)
        current_onset = None
        current_offset = None

        for item in group:
            onset = item.onset
            offset = item.offset

            if current_onset is None:
                current_onset = onset
                current_offset = offset
            elif onset <= current_offset + gap_threshold:
                # Extend current segment
                current_offset = max(current_offset, offset)
            else:
                # Save current merged event
                merged.append({
                    'filename': filename,
                    'onset': current_onset,
                    'offset': current_offset,
                    'event_label': label
                })
                current_onset = onset
                current_offset = offset

        # Save the last segment
        if current_onset is not None:
            merged.append({
                'filename': filename,
                'onset': current_onset,
                'offset': current_offset,
                'event_label': label
            })

    return merged


def evaluate(reference_csv, system_csv, gap_threshold=0.2):
    # Load reference and system annotations
    reference = MetaDataContainer().load(reference_csv)
    system_output = MetaDataContainer().load(system_csv)

    # Merge system output to reduce over-segmentation
    #system_output_merged = merge_events(system_output, gap_threshold=gap_threshold)
    system_output_merged = system_output

    # Initialize metrics with all labels
    labels = reference.unique_event_labels
    metrics = EventBasedMetrics(event_label_list=labels, t_collar= 0.75, percentage_of_length=0.5)

    # Evaluate per audio file
    for filename in reference.unique_files:
        ref_events = reference.filter(filename=filename)
        sys_events = system_output_merged.filter(filename=filename)
        metrics.evaluate(reference_event_list=ref_events, estimated_event_list=sys_events)

    print("Event-Based Metrics (after merging system events):")
    print(metrics.results_overall_metrics())


if __name__ == "__main__":
    # Set your paths
    reference_csv_path = "call_unit_events.csv"
    system_csv_path = "model_annotations_sed_eval.csv"

    # Evaluate
    evaluate(reference_csv_path, system_csv_path, gap_threshold=0.2)