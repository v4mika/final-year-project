import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import seaborn as sns

from data_utils import *
from pathlib import Path
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from umap import UMAP

# model_path = 'tested_models/final_models/cf_and_real_resnet18_bs128_lr0.0003_epochs5'

attributes = [
        'Manufacturer', 'scanner_type', 
        'race', 'is_positive', 'density', 'correct',
        'density_binary', 'age_group', 'prediction',
    ]

# attributes = ['scanner_type']

models = [
    'final_models_2/real_only_resnet50_bs64_lr0.0001_epochs10',
    'final_models_2/cf_and_real_resnet50_bs64_lr0.0001_epochs10',
    'final_models_2/cf_only_resnet50_bs128_lr0.0003_epochs10'
]

def get_embed_df():
    df = pd.read_csv('data/embed-non-negative.csv', low_memory=False)
    # df = pd.read_csv("/vol/biomedic3/mb121/tech-demo/code_for_demo/joined_simple.csv", low_memory=False)
    df['image_id'] = [img_path.split('/')[-1] for img_path in df.image_path.values]
    
    return df
        

def get_prd_emb_df(model_path, same_class=False):
    pred = 'predictions.csv'
    df_prd = pd.read_csv(os.path.join(model_path, pred))
    # df_prd = pd.read_csv(os.path.join(model_path, 'vindr_test.csv'))
    df_emb = pd.read_csv(os.path.join(model_path, 'embeddings.csv'))
    
    return df_prd, df_emb

modelname_group_map = {
    "Selenia Dimensions": "Selenia Dimensions",
    "Senographe Essential VERSION ADS_53.40": "Senographe Essential",
    "Senographe Essential VERSION ADS_54.10": "Senographe Essential",
    "Senograph 2000D ADS_17.4.5": "Senograph 2000D",
    "Senograph 2000D ADS_17.5": "Senograph 2000D",
    "Lorad Selenia": "Lorad Selenia",
    "Clearview CSm": "Clearview CSm",
    "Senographe Pristina": "Senographe Pristina",
}

def get_df_combined(df, df_prd, df_emb):
    df_combined = df_emb.merge(df_prd, on='image_id')
    df_combined['base_image_id'] = df_combined['image_id'].apply(lambda x: x.split('/')[0])

    pd.set_option('display.max_colwidth', None)

    df = df_combined.merge(df, left_on='base_image_id', right_on='image_id')
    
    # grouping together scanner name
    df['scanner_type'] = df['ManufacturerModelName'].map(modelname_group_map)
    
    # adding a prediction and correct columns
    df['prediction'] = df[['class_0', 'class_1']].idxmax(axis=1).apply(lambda x: int(x[-1]))
    df['density_binary'] = df['density'].apply(lambda x: 0 if x in ['A', 'B'] else 1)
    df['correct'] = (df['prediction'] == df['density_binary']).astype(int)
    
    bins = [30, 40, 50, 60, 70, 80, 90]
    labels = ['30-39', '40-49', 
            '50-59', '60-69', '70-79', '80-89']
    df['age_group'] = pd.cut(df['age_at_study'], bins=bins, labels=labels)
    df['age_group'] = pd.Categorical(df['age_group'], categories=labels, ordered=True)
        
    return df

def get_combined_df(model_path, same_class=False):
    df = get_embed_df()
    df_prd, df_emb = get_prd_emb_df(model_path, same_class)

    df_combined = df_emb.merge(df_prd, on='image_id')
    df_combined['base_image_id'] = df_combined['image_id'].apply(lambda x: x.split('/')[0])

    df = df_combined.merge(df, left_on='base_image_id', right_on='image_id')

    # adding a prediction and correct columns
    df['prediction'] = df[['class_0', 'class_1']].idxmax(axis=1).apply(lambda x: int(x[-1]))
    df['density_binary'] = df['density'].apply(lambda x: 0 if x in ['A', 'B'] else 1)
    df['correct'] = (df['prediction'] == df['density_binary']).astype(int)
    
    # grouping together scanner name
    df['scanner_type'] = df['ManufacturerModelName'].map(modelname_group_map)
    
    # adding age group bins
    bins = [30, 40, 50, 60, 70, 80, 90]
    labels = ['30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    df['age_group'] = pd.cut(df['age_at_study'], bins=bins, labels=labels, right=False)
    df['age_group'] = pd.Categorical(df['age_group'], categories=labels, ordered=True)

    # print(df.head)
    return df

def get_embeddings(df, num_features=512):
    embeddings = np.array(df.iloc[:,0:num_features])
    return embeddings

def get_silhouette_score(model_path, attribute, same_class=False):
    df = get_combined_df(model_path, same_class)
    
    num_features = 512
    
    embedding_columns = [str(x) for x in range(512)]
    X = df[embedding_columns].values

    # y = subgroup labels (here: race)
    y = df[attribute].values
    score = silhouette_score(X, y)
    print(f"Getting silhouette score for {model_path}")
    print(f"Overall silhouette score by {attribute}: {score:.3f}")
    silhouette_values = silhouette_samples(X, y)

    # Put into dataframe for per-group analysis
    df_silhouette = pd.DataFrame({
        attribute: y,
        'silhouette_value': silhouette_values
    })

    # Compute mean silhouette score per race group
    group_scores = df_silhouette.groupby(attribute)['silhouette_value'].mean().reset_index()

    # Display result
    print(group_scores)

def get_umap(model_path):
    df = get_combined_df(model_path)
    embeddings = get_embeddings(df)
    
    pca = decomposition.PCA(n_components=0.95, whiten=False)
    embeddings_pca = pca.fit_transform(embeddings)
        
    umap_model = UMAP(n_components=2, random_state=42)  # projects model into 2d
    embeddings_umap = umap_model.fit_transform(embeddings_pca)

    df['UMAP 1'] = embeddings_umap[:, 0]
    df['UMAP 2'] = embeddings_umap[:, 1]

    x = 'UMAP 1'
    y = 'UMAP 2'
    
    model_path = Path(model_path)
    model_name = model_path.parts[-1]

    output_dir = os.path.join('cf_only_umap_plots', model_name)
    os.makedirs(output_dir, exist_ok=True)  

    def plot_scatter(data, hue, x, y, palette):
        hue_order = list(data[hue].unique())
        hue_order.sort()
        sns.set_theme(style="white")
        ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, alpha=0.7, marker='o', s=20, palette=palette)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        return ax.get_figure()  # return figure to save it

    def plot_joint(data, hue, x, y, palette):
        hue_order = list(data[hue].unique())
        hue_order.sort()
        sns.set_theme(style="white")
        g = sns.jointplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, alpha=0.7, marker='o', s=20, palette=palette, marginal_kws={'common_norm': False})
        sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1.2, 1))
        return g.fig  # return figure to save it

    def plot_attribute(attribute):
        print(df[attribute].value_counts(normalize=False))
        print('')
        print(df[attribute].value_counts(normalize=True))

        num_colors = df[attribute].nunique()
        # color_palette = sns.color_palette("hsv", num_colors)
        color_palette = sns.color_palette("colorblind")

        fig = plot_joint(df, attribute, x, y, color_palette)

        # Save the figure
        save_path = os.path.join(output_dir, f"umap_{attribute}.png")
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close figure to avoid memory buildup

    # Iterate over attributes and plot + save
    for attribute in attributes:
        plot_attribute(attribute)

def compare_two_umaps(model_path_1, model_path_2, attributes):
    # Load first model
    df1 = get_combined_df(model_path_1)
    embeddings1 = get_embeddings(df1)
    
    pca1 = decomposition.PCA(n_components=0.95, whiten=False)
    embeddings_pca1 = pca1.fit_transform(embeddings1)
        
    umap_model1 = UMAP(n_components=2, random_state=42)
    embeddings_umap1 = umap_model1.fit_transform(embeddings_pca1)

    df1['UMAP 1'] = embeddings_umap1[:, 0]
    df1['UMAP 2'] = embeddings_umap1[:, 1]

    model_name1 = Path(model_path_1).parts[-1]

    # Load second model
    df2 = get_combined_df(model_path_2)
    embeddings2 = get_embeddings(df2)
    
    pca2 = decomposition.PCA(n_components=0.95, whiten=False)
    embeddings_pca2 = pca2.fit_transform(embeddings2)
        
    umap_model2 = UMAP(n_components=2, random_state=42)
    embeddings_umap2 = umap_model2.fit_transform(embeddings_pca2)

    df2['UMAP 1'] = embeddings_umap2[:, 0]
    df2['UMAP 2'] = embeddings_umap2[:, 1]

    model_name2 = Path(model_path_2).parts[-1]

    # Output dir
    output_dir = f'compare_umap_plots/{model_name1}_vs_{model_name2}'
    os.makedirs(output_dir, exist_ok=True)

    # Plot per attribute
    for attribute in attributes:
        print(f'Plotting {attribute}...')
        
        get_silhouette_score(model_path_1, attribute)
        get_silhouette_score(model_path_2, attribute)

        # Get color palette
        num_colors = max(df1[attribute].nunique(), df2[attribute].nunique())
        color_palette = sns.color_palette("colorblind", num_colors)

        hue_order = sorted(list(set(df1[attribute].unique()).union(set(df2[attribute].unique()))))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

        # Plot first model
        sns.scatterplot(
            data=df1, x='UMAP 1', y='UMAP 2',
            hue=attribute, hue_order=hue_order,
            palette=color_palette, alpha=0.7, ax=axes[0]
        )
        axes[0].set_title(f'real_only')

        # Plot second model
        sns.scatterplot(
            data=df2, x='UMAP 1', y='UMAP 2',
            hue=attribute, hue_order=hue_order,
            palette=color_palette, alpha=0.7, ax=axes[1]
        )
        axes[1].set_title(f'cf_only') # TODO need to change this

        # Remove duplicate legend from second plot
        axes[1].legend_.remove()

        # Shared legend below
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

        # Save
        save_path = os.path.join(output_dir, f'compare_umap_{attribute}.png')
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    print('All UMAP comparisons done.')
    
def compare_three_umaps(model_path_1, model_path_2, model_path_3, attributes):
    def process_model(model_path, same_class=False):
        df = get_combined_df(model_path, same_class)
        embeddings = get_embeddings(df)
        
        pca = decomposition.PCA(n_components=0.95, whiten=False)
        embeddings_pca = pca.fit_transform(embeddings)
        
        umap_model = UMAP(n_components=2, random_state=42)
        embeddings_umap = umap_model.fit_transform(embeddings_pca)
        
        df['UMAP 1'] = embeddings_umap[:, 0]
        df['UMAP 2'] = embeddings_umap[:, 1]
        
        return df, Path(model_path).parts[-1]

    # Process all 3 models
    df1, model_name1 = process_model(model_path_1)
    df2, model_name2 = process_model(model_path_2)
    df3, model_name3 = process_model(model_path_3, same_class=True)

    # Output dir
    output_dir = f'compare_umap_plots/{model_name1}_vs_{model_name2}_vs_{model_name3}'
    os.makedirs(output_dir, exist_ok=True)

    # Plot per attribute
    for attribute in attributes:
        print(f'Plotting {attribute}...')

        # Optionally print silhouette scores
        get_silhouette_score(model_path_1, attribute)
        get_silhouette_score(model_path_2, attribute)
        get_silhouette_score(model_path_3, attribute, same_class=False)

        # Get color palette
        num_colors = max(df1[attribute].nunique(), df2[attribute].nunique(), df3[attribute].nunique())
        color_palette = sns.color_palette("colorblind", num_colors)

        hue_order = sorted(list(
            set(df1[attribute].unique()).union(set(df2[attribute].unique())).union(set(df3[attribute].unique()))
        ))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

        # Plot model 1
        sns.scatterplot(
            data=df1, x='UMAP 1', y='UMAP 2',
            hue=attribute, hue_order=hue_order,
            palette=color_palette, alpha=0.7, ax=axes[0]
        )
        
        axes[0].set_title(f'real_only')
        
        # axes[0].set_title(f'{model_name1}')

        # Plot model 2
        sns.scatterplot(
            data=df2, x='UMAP 1', y='UMAP 2',
            hue=attribute, hue_order=hue_order,
            palette=color_palette, alpha=0.7, ax=axes[1]
        )
        axes[1].set_title('cf_and_real')
        # axes[1].set_title(f'{model_name2}')
        axes[1].legend_.remove()

        # Plot model 3
        sns.scatterplot(
            data=df3, x='UMAP 1', y='UMAP 2',
            hue=attribute, hue_order=hue_order,
            palette=color_palette, alpha=0.7, ax=axes[2]
        )
        axes[2].set_title('same_class_cf')
        # axes[2].set_title(f'{model_name3}')
        axes[2].legend_.remove()

        # Shared legend below
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))

        # Save
        save_path = os.path.join(output_dir, f'compare_umap_{attribute}.png')
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    print('All UMAP comparisons done.')

        
if __name__ == '__main__':
    vindr_cf_model = 'final_models_2/cf_only_resnet50_bs256_lr0.0005_epochs10'
    vindr_cf_and_real_model = 'final_models_2/cf_and_real_resnet50_bs64_lr0.0001_epochs10'
    vindr_real_model = 'final_models_2/real_only_resnet50_bs256_lr0.0005_epochs10'
    
    compare_three_umaps(vindr_real_model, vindr_cf_and_real_model, vindr_cf_model, attributes)
    
    # for model_path in models:
    #     get_umap(model_path)
    
    # for model_path in models:
    #     print(f'printing silhouette scores for model: {model_path}')
    #     for a in attributes:
    #         get_silhouette_score(model_path, a)
    
    
    # real_model_path = 'final_models_2/real_only_resnet50_bs64_lr0.0001_epochs10'
    # cf_model_path = 'final_models_2/cf_only_resnet50_bs128_lr0.0003_epochs10'
    # same_class_cf_model_path = 'same_class_models/same-class-cf_resnet50_bs256_lr0.0005_epochs10'

    
    # compare_three_umaps(real_model_path, cf_model_path, same_class_cf_model_path, attributes)
    
    # compare_two_umaps(real_model_path, cf_model_path, attributes)
    
    # cf_and_real_model_path = 'final_models_2/cf_and_real_resnet50_bs64_lr0.0001_epochs10'
    # compare_two_umaps(real_model_path, cf_and_real_model_path, attributes)

        