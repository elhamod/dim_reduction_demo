import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import numpy as np
import torch


from pythae.models import VAE, VAEConfig
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.pipelines import TrainingPipeline

# Shared colors
ORIG_COLOR = "royalblue"
RECON_COLOR = "darkorange"
ERROR_COLOR = "red"


# ============================================================
# PCA utilities
# ============================================================

def run_pca(X, n_pcs):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_pcs)
    scores = pca.fit_transform(X_scaled)
    X_scaled_recon = pca.inverse_transform(scores)
    X_recon = scaler.inverse_transform(X_scaled_recon)
    return pca, scaler, scores, X_recon


def make_3d_pca_geometry(
    X,
    pca,
    scaler,
    feature_names,
    pcs_to_show,
    selected_features,
    scaled_space: bool = False,
):
    """
    Unified 3D PCA geometry plot.

    If scaled_space == False:
        - Show everything in ORIGINAL feature units.
    If scaled_space == True:
        - Show everything in STANDARDIZED (scaled) space where
          projection is truly orthogonal and subspace is aligned.
    """
    idxs = [feature_names.index(f) for f in selected_features]

    # ---- Workhorse: always do PCA in scaled space ----
    X_scaled = scaler.transform(X)                      # (n, d)
    scores = pca.transform(X_scaled)                    # (n, k)
    X_scaled_recon = pca.inverse_transform(scores)      # (n, d)

    # Also compute original-space versions for errors / original view
    X_orig = scaler.inverse_transform(X_scaled)         # (n, d)
    X_recon_orig = scaler.inverse_transform(X_scaled_recon)

    # Decide what to actually PLOT
    if scaled_space:
        X_plot = X_scaled
        X_recon_plot = X_scaled_recon
        axis_suffix = " (scaled)"
        title_suffix = "PCA geometry in SCALED space"
    else:
        X_plot = X_orig
        X_recon_plot = X_recon_orig
        axis_suffix = ""
        title_suffix = "PCA projection & reconstruction in ORIGINAL space"

    X3 = X_plot[:, idxs]
    X3_recon = X_recon_plot[:, idxs]

    n = X3.shape[0]

    # full-space error (in ORIGINAL space, so it's invariant to the view)
    full_err = np.linalg.norm(X_orig - X_recon_orig, axis=1)

    import plotly.graph_objects as go
    fig = go.Figure()

    # ---------- Original & reconstructed points ----------
    def point_hover(i, kind):
        coords = "<br>".join(
            f"{selected_features[j]}{axis_suffix} = {X3[i, j]:.3f}" for j in range(3)
        )
        rcoords = "<br>".join(
            f"{selected_features[j]}{axis_suffix} = {X3_recon[i, j]:.3f}"
            for j in range(3)
        )
        return (
            f"{kind} point #{i}<br>"
            + "Plotted coords (subset):<br>"
            + coords
            + "<br><br>Reconstructed (subset):<br>"
            + rcoords
            + f"<br><br>Total reconstruction error (L2, original {X.shape[1]}D) = {full_err[i]:.4f}"
        )

    # original points
    fig.add_trace(
        go.Scatter3d(
            x=X3[:, 0],
            y=X3[:, 1],
            z=X3[:, 2],
            mode="markers",
            marker=dict(size=6, color=ORIG_COLOR),
            name=f"Original{axis_suffix}",
            hovertext=[point_hover(i, "Original") for i in range(n)],
            hoverinfo="text",
        )
    )


    # reconstructed points
    fig.add_trace(
        go.Scatter3d(
            x=X3_recon[:, 0],
            y=X3_recon[:, 1],
            z=X3_recon[:, 2],
            mode="markers",
            marker=dict(size=4, symbol="x", color=RECON_COLOR),
            name=f"Reconstructed{axis_suffix}",
            hovertext=[point_hover(i, "Reconstructed") for i in range(n)],
            hoverinfo="text",
        )
    )

    # ---------- Error segments in the plotted space ----------
    for i in range(n):
        fig.add_trace(
            go.Scatter3d(
                x=[X3[i, 0], X3_recon[i, 0]],
                y=[X3[i, 1], X3_recon[i, 1]],
                z=[X3[i, 2], X3_recon[i, 2]],
                mode="lines",
                line=dict(width=2, color=ERROR_COLOR),
                showlegend=(i == 0),
                name="Reconstruction error (plotted space)",
                hoverinfo="text",
                hovertext=(
                    f"Error segment for point #{i}<br>"
                    f"L2 error (3D plotted space) = {np.linalg.norm(X3[i] - X3_recon[i]):.4f}<br>"
                    f"L2 error (original {X.shape[1]}D) = {full_err[i]:.4f}"
                ),
            )
        )

    # ---------- Projection line / plane (always built in SCALED space) ----------
    d = X.shape[1]
    origin_scaled = np.zeros((1, d))     # PCA subspace passes through 0 in scaled space

    # Use score ranges to size line/plane
    if scores.shape[1] >= 1:
        s1_min, s1_max = scores[:, 0].min(), scores[:, 0].max()
    else:
        s1_min, s1_max = -3.0, 3.0

    if pcs_to_show == 1 and pca.components_.shape[0] >= 1:
        pc1 = pca.components_[0]  # unit vector in scaled space

        t_vals = np.linspace(s1_min * 1.2, s1_max * 1.2, 40)
        line_scaled = origin_scaled + np.outer(t_vals, pc1)  # (40, d)

        if scaled_space:
            line_plot_full = line_scaled
        else:
            line_plot_full = scaler.inverse_transform(line_scaled)

        line3 = line_plot_full[:, idxs]

        fig.add_trace(
            go.Scatter3d(
                x=line3[:, 0],
                y=line3[:, 1],
                z=line3[:, 2],
                mode="lines",
                line=dict(width=4, color="black"),
                name=f"PC1 line{axis_suffix}",
                hoverinfo="skip",
            )
        )

    elif pcs_to_show == 2 and pca.components_.shape[0] >= 2:
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]

        s1_vals = np.linspace(scores[:, 0].min() * 1.2, scores[:, 0].max() * 1.2, 25)
        s2_vals = np.linspace(scores[:, 1].min() * 1.2, scores[:, 1].max() * 1.2, 25)
        S1, S2 = np.meshgrid(s1_vals, s2_vals)

        grid_flat = np.stack([S1.ravel(), S2.ravel()], axis=1)  # (N, 2)

        plane_scaled = (
            origin_scaled
            + grid_flat[:, 0:1] * pc1
            + grid_flat[:, 1:2] * pc2
        )  # (N, d)

        if scaled_space:
            plane_plot_full = plane_scaled
        else:
            plane_plot_full = scaler.inverse_transform(plane_scaled)

        plane3 = plane_plot_full[:, idxs].reshape(S1.shape[0], S1.shape[1], 3)

        fig.add_trace(
            go.Surface(
                x=plane3[:, :, 0],
                y=plane3[:, :, 1],
                z=plane3[:, :, 2],
                opacity=0.35,
                showscale=False,
                name=f"PC1–PC2 plane{axis_suffix}",
                hoverinfo="skip",
                surfacecolor=np.zeros_like(plane3[:, :, 0]),      # ← CONSTANT VALUE
                colorscale=[[0, "black"], [1, "black"]],          # ← FLAT COLOR
            )
        )


    fig.update_layout(
        scene=dict(
            xaxis_title=selected_features[0] + axis_suffix,
            yaxis_title=selected_features[1] + axis_suffix,
            zaxis_title=selected_features[2] + axis_suffix,
            aspectmode="data",           # 🔥 SAME SCALE ON ALL AXES
        ),
        title=title_suffix,
        height=750,
    )
    return fig


# def make_pca_score_figure(scores, pcs_to_show):
#     n_samples, n_components = scores.shape
#     pcs_to_show = min(pcs_to_show, n_components)

#     if pcs_to_show == 1:
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=np.arange(n_samples),
#                 y=scores[:, 0],
#                 mode="markers+lines",
#                 name="PC1 score",
#                 hoverinfo="text",
#                 hovertext=[
#                     f"Point #{i}<br>PC1 score = {scores[i,0]:.4f}"
#                     for i in range(n_samples)
#                 ],
#             )
#         )
#         fig.update_layout(
#             xaxis_title="Sample index",
#             yaxis_title="PC1 score",
#             title="PCA scores (1D)",
#             height=500,
#         )

#     elif pcs_to_show == 2:
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=scores[:, 0],
#                 y=scores[:, 1],
#                 mode="markers",
#                 name="Scores",
#                 hoverinfo="text",
#                 hovertext=[
#                     f"Point #{i}<br>PC1 = {scores[i,0]:.4f}<br>PC2 = {scores[i,1]:.4f}"
#                     for i in range(n_samples)
#                 ],
#             )
#         )
#         fig.update_layout(
#             xaxis_title="PC1",
#             yaxis_title="PC2",
#             title="PCA scores (2D)",
#             height=500,
#         )

#     else:
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter3d(
#                 x=scores[:, 0],
#                 y=scores[:, 1],
#                 z=scores[:, 2],
#                 mode="markers",
#                 marker=dict(size=5),
#                 name="Scores",
#                 hoverinfo="text",
#                 hovertext=[
#                     f"Point #{i}<br>"
#                     f"PC1 = {scores[i,0]:.4f}<br>"
#                     f"PC2 = {scores[i,1]:.4f}<br>"
#                     f"PC3 = {scores[i,2]:.4f}"
#                     for i in range(n_samples)
#                 ],
#             )
#         )
#         fig.update_layout(
#             scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
#             title="PCA scores (3D)",
#             height=700,
#         )

#     return fig


# ============================================================
# Pythae VAE utilities
# ============================================================
from typing import List
import torch.nn as nn
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.base.base_utils import ModelOutput

layer_1_size = 32
# class Encoder_AE_MLP(BaseEncoder):
#     def __init__(self, args: dict):
#         BaseEncoder.__init__(self)
#         self.input_dim = args.input_dim
#         self.latent_dim = args.latent_dim
    
#         layers = nn.ModuleList()

#         layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), layer_1_size), nn.ReLU()))

#         self.layers = layers
#         self.depth = len(layers)

#         self.embedding = nn.Linear(layer_1_size, self.latent_dim)

#     def forward(self, x, output_layer_levels: List[int] = None):
#         output = ModelOutput()

#         max_depth = self.depth

#         out = x.reshape(-1, np.prod(self.input_dim))

#         for i in range(max_depth):
#             out = self.layers[i](out)

#             if output_layer_levels is not None:
#                 if i + 1 in output_layer_levels:
#                     output[f"embedding_layer_{i+1}"] = out
#             if i + 1 == self.depth:
#                 output["embedding"] = self.embedding(out)

#         return output




class Encoder_VAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), layer_1_size), nn.Tanh()))
        layers.append(nn.Sequential(nn.Linear(layer_1_size, layer_1_size), nn.Tanh()))
        # layers.append(nn.Sequential(nn.Linear(layer_1_size, layer_1_size), nn.Tanh()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(layer_1_size, self.latent_dim)
        self.log_var = nn.Linear(layer_1_size, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)

        return output



class Decoder_AE_MLP(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, layer_1_size), nn.Tanh()))
        layers.append(nn.Sequential(nn.Linear(layer_1_size, layer_1_size), nn.Tanh()))
        # layers.append(nn.Sequential(nn.Linear(layer_1_size, layer_1_size), nn.Tanh()))

        layers.append(
           nn.Linear(layer_1_size, int(np.prod(args.input_dim)))
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth
        
        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output



class StreamlitLossCallback(TrainingCallback):
    """
    Simple Pythae callback that updates a Streamlit placeholder with
    the latest training / eval loss every time logs are emitted.
    """

    def __init__(self, loss_placeholder):
        super().__init__()
        self.loss_placeholder = loss_placeholder
        self.best_train_loss = np.inf
        self.num_epochs = 0

    def on_log(self, training_config, logs, **kwargs):
        # print("hi", kwargs)
        self.num_epochs = self.num_epochs + 1
        
        # print("my logs", logs)
        # train_loss = logs.get("train_loss", None)
        # eval_loss = logs.get("eval_loss", None)
        # epoch = logs.get("epoch", None)

        # parts = []
        # if epoch is not None:
        #     parts.append(f"Epoch: {epoch}")
        # if train_loss is not None:
        #     parts.append(f"train_loss = {metrics["train_epoch_loss]:.4f}")
        # if eval_loss is not None:
        #     parts.append(f"eval_loss = {eval_loss:.4f}")

        # if parts:
        loss = logs["train_epoch_loss"]
        if loss < self.best_train_loss:
            self.best_train_loss = loss
        if self.num_epochs % 100 == 0:
            self.loss_placeholder.markdown(f"**Epoch** = {kwargs["global_step"]}. **VAE training loss** = {loss:.4f}. **Best training loss** = {self.best_train_loss:.4f}")


def train_pythae_vae(
    X: np.ndarray,
    latent_dim: int = 2,
    num_epochs: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    # output_dir: str = "pythae_vae_runs",
    loss_callback: TrainingCallback | None = None,
):
    """
    Train a Pythae VAE on tabular data X (n_samples x n_features).

    Returns
    -------
    model : VAE
        Trained VAE model (can be used for decoding a latent grid).
    Z : np.ndarray
        Latent embeddings of shape (n_samples, latent_dim).
    X_recon : np.ndarray
        Reconstructions of X, same shape as X.
    """

    # Pythae expects float32 tensors
    X = X.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_features = X.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build VAE model config for tabular data
    model_config = VAEConfig(
        input_dim=(n_features,),   # 1D vector input
        latent_dim=latent_dim
    )
    model = VAE(model_config, encoder=Encoder_VAE_MLP(model_config), decoder=Decoder_AE_MLP(model_config)).to(device)
    # model_config = AEConfig(
    #     input_dim=(n_features,),   # 1D vector input
    #     latent_dim=latent_dim
    # )
    # model = AE(model_config, encoder=Encoder_AE_MLP(model_config), decoder=Decoder_AE_MLP(model_config)).to(device)

    # print(model)
    
    # 2) Trainer config (this is the correct class in 0.1.2)
    train_config = BaseTrainerConfig(
        # output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=min(batch_size, len(X)),
        per_device_eval_batch_size=min(batch_size, len(X)),
        steps_saving=None,
        steps_predict=None,
        no_cuda=(device == "cpu"),
        keep_best_on_train=True,
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 2000, "factor": 0.9},
        # scheduler_cls="MultiStepLR",
        # scheduler_params={
        #     "milestones": [200, 350, 500, 750, 1000],
        #     "gamma": 10 ** (-1 / 5),
        #     # "verbose": True,
        # },
        optimizer_cls="RMSprop",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)}
    )

    # 3) Training pipeline (handles DataProcessor / BaseDataset internally)
    pipeline = TrainingPipeline(
        model=model,
        training_config=train_config
    )
    
    callbacks = []
    if loss_callback is not None:
        callbacks.append(loss_callback)
        
    # Train directly from numpy array
    # print(X_scaled.shape, X_scaled)
    # raise
    pipeline(train_data=X_scaled, callbacks=callbacks)

    # After training, the trained model is stored in pipeline.model
    trained_model = pipeline.model.to(device)
    trained_model.eval()

    # 4) Get reconstructions & latent embeddings using .predict
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        out = trained_model.predict(X_tensor)

        # According to BaseAE.predict, keys are recon_x and embedding
        # https://pythae.readthedocs.io/en/latest/_modules/pythae/models/base/base_model.html
        X_recon = out.recon_x.detach().cpu().numpy()
        Z = out.embedding.detach().cpu().numpy()

    return trained_model, Z, scaler.inverse_transform(X_recon)



def make_vae_latent_and_manifold_figures(
    model, Z, X, X_recon_vae, feature_names, selected_features, grid_points=15
):
    """
    - Latent scatter (1D or 2D)
    - Manifold: line (1D latent) or surface (2D latent) in original coordinates
    - Uses PCA colors for datapoints & black for VAE geometry
    - Shows projection errors (original -> VAE recon) in original space
    """
    device = next(model.parameters()).device
    idxs = [feature_names.index(f) for f in selected_features]
    latent_dim = Z.shape[1]

    # ---------- Latent scatter ----------
    if latent_dim == 1:
        fig_latent = go.Figure()
        fig_latent.add_trace(
            go.Scatter(
                x=np.arange(Z.shape[0]),
                y=Z[:, 0],
                mode="markers+lines",
                line=dict(color="black"),
                marker=dict(color="black"),
                name="Latent points",
                hoverinfo="text",
                hovertext=[
                    f"Point #{i}<br>z1 = {Z[i,0]:.4f}" for i in range(Z.shape[0])
                ],
            )
        )
        fig_latent.update_layout(
            xaxis_title="Sample index",
            yaxis_title="z1",
            title="VAE latent space (1D)",
            height=500,
        )
    else:
        fig_latent = go.Figure()
        fig_latent.add_trace(
            go.Scatter(
                x=Z[:, 0],
                y=Z[:, 1],
                mode="markers",
                marker=dict(color="black"),
                name="Latent points",
                hoverinfo="text",
                hovertext=[
                    f"Point #{i}<br>z1 = {Z[i,0]:.4f}<br>z2 = {Z[i,1]:.4f}"
                    for i in range(Z.shape[0])
                ],
            )
        )
        fig_latent.update_layout(
            xaxis_title="z1",
            yaxis_title="z2",
            title="VAE latent space (2D)",
            height=500,
        )

    # ---------- Manifold & projection errors in original space ----------
    X3_orig = X[:, idxs]
    X3_recon = X_recon_vae[:, idxs]

    fig_manifold = go.Figure()

    # Original points (same color as PCA)
    fig_manifold.add_trace(
        go.Scatter3d(
            x=X3_orig[:, 0],
            y=X3_orig[:, 1],
            z=X3_orig[:, 2],
            mode="markers",
            marker=dict(size=6, color=ORIG_COLOR),
            name="Original (VAE view)",
            hoverinfo="text",
            hovertext=[
                f"Point #{i}<br>"
                + "<br>".join(
                    f"{selected_features[j]} = {X3_orig[i, j]:.4f}" for j in range(3)
                )
                for i in range(X3_orig.shape[0])
            ],
        )
    )

    # VAE recon points (same color as PCA recon)
    fig_manifold.add_trace(
        go.Scatter3d(
            x=X3_recon[:, 0],
            y=X3_recon[:, 1],
            z=X3_recon[:, 2],
            mode="markers",
            marker=dict(size=4, symbol="x", color=RECON_COLOR),
            name="VAE recon points",
            hoverinfo="text",
            hovertext=[
                f"Decoded point #{i}<br>"
                + "<br>".join(
                    f"{selected_features[j]} = {X3_recon[i, j]:.4f}" for j in range(3)
                )
                for i in range(X3_recon.shape[0])
            ],
        )
    )

    # Projection error lines (original -> VAE recon), black
    for i in range(X3_orig.shape[0]):
        fig_manifold.add_trace(
            go.Scatter3d(
                x=[X3_orig[i, 0], X3_recon[i, 0]],
                y=[X3_orig[i, 1], X3_recon[i, 1]],
                z=[X3_orig[i, 2], X3_recon[i, 2]],
                mode="lines",
                line=dict(width=2, color=ERROR_COLOR),
                showlegend=(i == 0),
                name="VAE reconstruction error",
                hoverinfo="skip",
            )
        )

    # Nonlinear manifold from latent grid (black)
    if latent_dim == 1:
        # 1D manifold: a curve in 3D
        z1_min, z1_max = Z[:, 0].min(), Z[:, 0].max()
        if z1_min == z1_max:
            z1_min -= 1.0
            z1_max += 1.0
        z1_lin = np.linspace(z1_min, z1_max, grid_points)
        grid_z = z1_lin.reshape(-1, 1)

        with torch.no_grad():
            z_tensor = torch.tensor(grid_z.astype(np.float32)).to(device)
            decoded_out = model.decoder(z_tensor)
            decoded = decoded_out["reconstruction"].cpu().numpy()

        decoded3 = decoded[:, idxs]
        fig_manifold.add_trace(
            go.Scatter3d(
                x=decoded3[:, 0],
                y=decoded3[:, 1],
                z=decoded3[:, 2],
                mode="lines",
                line=dict(width=3, color="black"),
                name="Decoded latent curve",
                hoverinfo="text",
                hovertext=(
                    "Nonlinear manifold: decoded VAE curve<br>"
                    "Each point comes from decoding a z1 value in latent space."
                ),
            )
        )
    else:
        # 2D manifold: a surface in 3D
        z1_min, z1_max = Z[:, 0].min(), Z[:, 0].max()
        z2_min, z2_max = Z[:, 1].min(), Z[:, 1].max()
        if z1_min == z1_max:
            z1_min -= 1.0
            z1_max += 1.0
        if z2_min == z2_max:
            z2_min -= 1.0
            z2_max += 1.0

        z1_lin = np.linspace(z1_min, z1_max, grid_points)
        z2_lin = np.linspace(z2_min, z2_max, grid_points)
        Z1, Z2 = np.meshgrid(z1_lin, z2_lin)
        grid_z = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

        with torch.no_grad():
            z_tensor = torch.tensor(grid_z.astype(np.float32)).to(device)
            decoded_out = model.decoder(z_tensor)
            decoded = decoded_out["reconstruction"].cpu().numpy()

        decoded3 = decoded[:, idxs].reshape(grid_points, grid_points, 3)

        fig_manifold.add_trace(
            go.Surface(
                x=decoded3[:, :, 0],
                y=decoded3[:, :, 1],
                z=decoded3[:, :, 2],
                opacity=0.5,
                showscale=False,
                name="Decoded latent grid",
                hoverinfo="text",
                hovertext=(
                    "Nonlinear manifold: decoded VAE grid<br>"
                    "Each surface point comes from decoding a regular grid in (z1, z2)."
                ),
                surfacecolor=np.zeros_like(decoded3[:, :, 0]),
                colorscale=[[0, "black"], [1, "black"]],
            )
        )

    fig_manifold.update_layout(
        scene=dict(
            xaxis_title=selected_features[0],
            yaxis_title=selected_features[1],
            zaxis_title=selected_features[2],
            aspectmode="data",           # 🔥 SAME SCALE ON ALL AXES
        ),
        title="VAE nonlinear manifold & reconstruction (original space)",
        height=750,
    )

    return fig_latent, fig_manifold



# ============================================================
# Streamlit app
# ============================================================

def main():
    st.set_page_config(
        page_title="PCA & VAE Playground (Pythae)",
        layout="wide",
    )

    st.title("PCA + VAE (Pythae) Interactive Playground")

    with st.expander("What this tool shows (for students / viewers)", expanded=False):
        st.markdown(
            """
- **PCA view**: linear projection onto principal components.
  - You see:
    - Original vs reconstructed points in original feature space.
    - **Error segments** from each point to its reconstruction.
    - A **projection line** (PC1) or **projection plane** (PC1–PC2).
- **VAE view (Pythae)**: nonlinear embedding.
  - Learns a 2D latent space \\((z_1, z_2)\\).
  - We take a **regular grid** in latent space, decode it back, and show the resulting
    **curved manifold surface** in the original feature coordinates.
            """
        )

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")

        uploaded_file = st.file_uploader(
            "Upload CSV (optional, numeric columns only)", type=["csv"]
        )

        # 2) If a CSV is uploaded, replace the table with its numeric columns
        if uploaded_file is not None: # and "csv_loaded" not in st.session_state:
            try:
                df_csv = pd.read_csv(uploaded_file)
                df_num = df_csv.select_dtypes(include="number")
        
                if df_num.shape[1] == 0:
                    st.warning("Uploaded CSV has no numeric columns; keeping existing table.")
                else:
                    # limit to first 10 numeric columns
                    if df_num.shape[1] > 10:
                        df_num = df_num.iloc[:, :10]
        
                    st.session_state.data_df = df_num.copy()
        
                    # 🔥 update num_features based on CSV
                    st.session_state.num_features = df_num.shape[1]
                    # st.session_state.csv_loaded = True
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        
        if "num_features" not in st.session_state:
            st.session_state.num_features = 3

        num_features = st.slider(
            "Number of features (dimensions)",
            1, 10,
            # value=st.session_state.num_features,
            key="num_features",     # ⬅ same key name
        )
        
        pcs_to_show = st.slider("Number of PCs to visualize", 1, 3, 2)
        show_scaled_space = st.checkbox("Show PCA geometry in scaled space", value=False)

        st.markdown("---")
        use_vae = st.checkbox("Enable VAE (Pythae nonlinear view)", value=True)
        vae_epochs = 200
        latent_dim = 2
        if use_vae:
            vae_epochs = st.slider("VAE training epochs", 500, 50000, 2500, step=500)
            latent_dim = st.radio("VAE latent dimension", [1, 2], index=1)
            st.caption("Latent dimension is fixed at 2 for 2D manifold visualization.")



    

    st.subheader("1. Enter / edit your data points")
    st.markdown(
        "Each **row** is a datapoint, each **column** is a feature. "
        "You can add or delete rows. All values are treated as numeric."
    )

    # --- persistent table state in main area ---
    # 1) Initialize if not present
    # num_features is whatever the slider currently says
    num_features = st.session_state.num_features
    
    # init data_df if missing
    if "data_df" not in st.session_state:
        st.session_state.data_df = pd.DataFrame(
            np.random.randn(6, num_features),
            columns=[f"x{i+1}" for i in range(num_features)],
        )
    else:
        df = st.session_state.data_df
        current_n = df.shape[1]
    
        if current_n != num_features:
            new_cols = [f"x{i+1}" for i in range(num_features)]
            new_df = pd.DataFrame(index=df.index)
    
            # keep any existing columns that still fit
            for c in new_cols:
                if c in df.columns:
                    new_df[c] = df[c]
                else:
                    new_df[c] = np.random.randn(len(df))  # random for new cols
    
            st.session_state.data_df = new_df


        


    # 4) Show editor using the stored dataframe
    edited_df = st.data_editor(
        st.session_state.data_df,
        key="data_editor",
        num_rows="dynamic",
        use_container_width=True,
    )

    # 5) Sync back user edits
    st.session_state.data_df = edited_df


    # 3) Now define feature_names and num_features from the CURRENT table
    feature_names = list(st.session_state.data_df.columns)
    num_features = len(feature_names)

    if "run_pca" not in st.session_state:
        st.session_state.run_pca = False
        

    if st.button("Run PCA (and VAE if enabled)", type="primary"):
        st.session_state.run_pca = True

    if st.session_state.run_pca:
        # ---------- Prepare X ----------
        try:
            X = edited_df[feature_names].astype(float).values
        except Exception as e:
            st.error(f"Could not parse table data as numeric: {e}")
            st.session_state.run_pca = False
            return
    
        n_samples = X.shape[0]
        if n_samples < 2:
            st.warning("Need at least 2 data points for PCA.")
            st.session_state.run_pca = False
            return
        else:
            try:
                X = edited_df[feature_names].astype(float).values
            except Exception as e:
                st.error(f"Could not parse table data as numeric: {e}")
                st.session_state.run_pca = False
                return


        # ====================================================
        # PCA
        # ====================================================
        pca, scaler, scores, X_recon = run_pca(X, pcs_to_show)

        # ====================================================
        # 2a. Show reconstructed values & PCA scores as tables
        # ====================================================
        st.subheader("2a. Reconstructed values and PCA scores")

        # Reconstructed values in original feature space
        recon_df = pd.DataFrame(X_recon, columns=feature_names)
        st.markdown("**Reconstructed values (in original feature units):**")
        st.dataframe(recon_df.style.format("{:.4f}"), use_container_width=True)

        # PCA scores (coordinates in PC space)
        score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
        scores_df = pd.DataFrame(scores, columns=score_cols)
        st.markdown("**PCA scores (projection coordinates in PC-space):**")
        st.dataframe(scores_df.style.format("{:.4f}"), use_container_width=True)


        st.subheader("2b. PCA components and explained variance")

        comp_labels = [f"PC{i+1}" for i in range(pca.components_.shape[0])]
        loadings = pd.DataFrame(
            pca.components_,
            index=comp_labels,
            columns=feature_names,
        )
        c1, c2 = st.columns((1.5, 1))
        with c1:
            st.markdown("**PCA loadings (components expressed in feature space):**")
            st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)

        with c2:
            evr = pca.explained_variance_ratio_
            evr_df = pd.DataFrame(
                {"PC": comp_labels, "Explained variance ratio": evr}
            )
            st.markdown("**Explained variance ratio:**")
            st.dataframe(evr_df.style.format({"Explained variance ratio": "{:.3f}"}))

        # ====================================================
        # Original-space 3D PCA visualization
        # ====================================================
        st.subheader("3. PCA projection in original feature space")

        if num_features >= 3:
            selected_features = st.multiselect(
                "Choose 3 features for the 3D axes (original space):",
                feature_names,
                default=feature_names[:3],
            )
            if len(selected_features) != 3:
                st.warning("Please select exactly 3 features.")
            else:
                fig_orig = make_3d_pca_geometry(
                        X,
                        pca,
                        scaler,
                        feature_names,
                        pcs_to_show,
                        selected_features,
                        scaled_space=show_scaled_space,
                    )

                st.plotly_chart(fig_orig, use_container_width=True)
        else:
            st.info(
                "You need at least 3 features to show a 3D original-space plot. "
                "Increase the number of features in the sidebar."
            )

        # ====================================================
        # PCA score space
        # ====================================================
        # st.subheader("4. PCA score space (projection coordinates)")
        # st.markdown(
        #     "Here we look at the coordinates of each point **in PC-space** rather than original feature space."
        # )
        # fig_scores = make_pca_score_figure(scores, pcs_to_show)
        # st.plotly_chart(fig_scores, use_container_width=True)

        # ====================================================
        # VAE (Pythae) nonlinear projection + manifold
        # ====================================================
        if use_vae:
            st.subheader("5. VAE nonlinear embedding (Pythae)")

            # placeholder for live loss updates
            training_status = st.empty()

            with st.spinner("Training a small VAE on your data..."):
                vae_callback = StreamlitLossCallback(training_status)
                
                model, Z, X_recon_vae = train_pythae_vae(
                    X,
                    latent_dim=latent_dim,
                    num_epochs = vae_epochs,
                    # batch_size=2,
                    learning_rate=1e-4,
                    # output_dir: str = "pythae_vae_runs",
                    loss_callback=vae_callback,
                )



            if num_features >= 3:
                selected_features_vae = st.multiselect(
                    "Choose 3 features for the VAE manifold in original space:",
                    feature_names,
                    default=feature_names[:3],
                    key="vae_axes",
                )
                if len(selected_features_vae) != 3:
                    st.warning("Please select exactly 3 features for the VAE manifold.")
                else:
                    fig_latent, fig_manifold = make_vae_latent_and_manifold_figures(
                        model,
                        Z,
                        X,
                        X_recon_vae,
                        feature_names,
                        selected_features_vae,
                        grid_points=18,
                    )

                    st.plotly_chart(fig_latent, use_container_width=True)
                    st.plotly_chart(fig_manifold, use_container_width=True)
            else:
                st.info(
                    "You need at least 3 features to show a 3D decoded manifold in original space."
                )


if __name__ == "__main__":
    main()
