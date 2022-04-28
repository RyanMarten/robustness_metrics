import matplotlib.pyplot as plt
import pandas as pd

def plot(df_main, ModelFamily='clip', ModelName='clip_vit_b32'):
    mask = df_main.ModelFamily == ModelFamily
    mask &= df_main.ModelName == ModelName
    mask &= df_main.Metric == 'accuracy'
    mask &= df_main.rescaling_method == 'none' 
    mask &= df_main.DatasetName == "imagenet(split='validation[20%:]')"
    imagenet_acc = df_main[mask].MetricValue.values[0]
    # print(f"CLIP ViT-B/32 ImageNet-Val Accuracy: {imagenet_acc}")
    
    mask = df_main.ModelFamily == ModelFamily
    mask &= df_main.ModelName == ModelName  # 'clip_r101', 'clip_r50', 'clip_r50x4'
    mask &= df_main.Metric == 'accuracy'
    mask &= df_main.rescaling_method == 'none' # temperature scaling only affects confidence, not accuracy
    mask &= df_main.DatasetName.str.contains('imagenet_c')
    df = df_main[mask]  # 75 rows = 15 distortions * 5 serverties
    
    mid = len(df.corruption_type.unique()) // 2
    distortions = df.corruption_type.unique()
    severities = df.severity.unique()

    for dist in [distortions[:mid], distortions[mid:]]:
        rows = []
        for distortion_name in dist:
                row = [distortion_name]
                for severity in severities:
                    mask = (df.severity == severity) & (df.corruption_type == distortion_name)
                    df[mask].MetricValue.values[0]
                    acc = df[mask].MetricValue.values[0]
                    row.append(acc)
                rows.append(row)

        columns = ["Method", "Severity 1", "Severity 2", "Severity 3", "Severity 4", "Severity 5"]
        p = pd.DataFrame(rows, columns=columns).plot(x='Method',
                kind='bar',
                stacked=False,
                rot=0,
                xlabel=" ",
                figsize=(7,1), # fontsize=12,
                ylim=(0.0,1.0));

        p.axhline(y=imagenet_acc, color='r', linestyle='--', label='No Distortion', linewidth=2);
        p.legend(bbox_to_anchor=(1.14,0.5), loc="right", borderaxespad=0); #, prop={'size': size});

        plt.rcParams["savefig.bbox"] = 'tight'
        fig = p.get_figure()
        fig.savefig('first.jpeg', dpi=300)