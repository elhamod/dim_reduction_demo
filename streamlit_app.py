import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from pythae.models import VAE, VAEConfig, AE, AEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.pipelines import TrainingPipeline
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.base.base_utils import ModelOutput


# Shared colors
ORIG_COLOR = "royalblue"
RECON_COLOR = "darkorange"
ERROR_COLOR = "red"


def _coerce_num_rows(default: int = 6, min_value: int = 1) -> int:
    raw = st.session_state.get("num_rows_text", str(default)).strip()
    try:
        n = int(raw)
    except ValueError:
        n = default
    return max(min_value, n)

# ============================================================
# PCA utilities
# ============================================================

def run_pca(X: np.ndarray, n_pcs: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_pcs)
    scores = pca.fit_transform(X_scaled)
    X_scaled_recon = pca.inverse_transform(scores)
    X_recon = scaler.inverse_transform(X_scaled_recon)
    return pca, scaler, scores, X_recon


def make_3d_pca_geometry(
    X: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    feature_names: list[str],
    pcs_to_show: int,
    selected_features: list[str],
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

    # ---- PCA always computed in scaled space ----
    X_scaled = scaler.transform(X)                      # (n, d)
    scores = pca.transform(X_scaled)                    # (n, k)
    X_scaled_recon = pca.inverse_transform(scores)      # (n, d)

    # Original-space versions for errors / original view
    X_orig = scaler.inverse_transform(X_scaled)         # (n, d)
    X_recon_orig = scaler.inverse_transform(X_scaled_recon)

    # Choose what to plot
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

    # Full-space error (in original space; invariant to the view)
    full_err = np.linalg.norm(X_orig - X_recon_orig, axis=1)

    fig = go.Figure()

    def point_hover(i: int, kind: str) -> str:
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

    # Original points
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

    # Reconstructed points
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

    # Error segments in plotted space
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

    # Projection line / plane (built in scaled space)
    d = X.shape[1]
    origin_scaled = np.zeros((1, d))

    s1_min, s1_max = (scores[:, 0].min(), scores[:, 0].max()) if scores.shape[1] >= 1 else (-3.0, 3.0)

    if pcs_to_show == 1 and pca.components_.shape[0] >= 1:
        pc1 = pca.components_[0]  # scaled space

        t_vals = np.linspace(s1_min * 1.2, s1_max * 1.2, 40)
        line_scaled = origin_scaled + np.outer(t_vals, pc1)

        line_plot_full = line_scaled if scaled_space else scaler.inverse_transform(line_scaled)
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
        grid_flat = np.stack([S1.ravel(), S2.ravel()], axis=1)

        plane_scaled = origin_scaled + grid_flat[:, 0:1] * pc1 + grid_flat[:, 1:2] * pc2
        plane_plot_full = plane_scaled if scaled_space else scaler.inverse_transform(plane_scaled)
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
                surfacecolor=np.zeros_like(plane3[:, :, 0]),
                colorscale=[[0, "black"], [1, "black"]],
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title=selected_features[0] + axis_suffix,
            yaxis_title=selected_features[1] + axis_suffix,
            zaxis_title=selected_features[2] + axis_suffix,
            aspectmode="data",
        ),
        title=title_suffix,
        height=750,
    )
    return fig


# ============================================================
# Pythae AE/VAE utilities
# ============================================================

LAYER_1_SIZE = 32


class Encoder_AE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        super().__init__()
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(int(np.prod(args.input_dim)), LAYER_1_SIZE), nn.Tanh()),
                nn.Sequential(nn.Linear(LAYER_1_SIZE, LAYER_1_SIZE), nn.Tanh()),
            ]
        )
        self.embedding = nn.Linear(LAYER_1_SIZE, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()
        out = x.reshape(-1, int(np.prod(self.input_dim)))

        for i, layer in enumerate(self.layers, start=1):
            out = layer(out)
            if output_layer_levels is not None and i in output_layer_levels:
                output[f"embedding_layer_{i}"] = out

        output["embedding"] = self.embedding(out)
        return output


class Encoder_VAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        super().__init__()
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(int(np.prod(args.input_dim)), LAYER_1_SIZE), nn.Tanh()),
                nn.Sequential(nn.Linear(LAYER_1_SIZE, LAYER_1_SIZE), nn.Tanh()),
            ]
        )
        self.embedding = nn.Linear(LAYER_1_SIZE, self.latent_dim)
        self.log_var = nn.Linear(LAYER_1_SIZE, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()
        out = x.reshape(-1, int(np.prod(self.input_dim)))

        for i, layer in enumerate(self.layers, start=1):
            out = layer(out)
            if output_layer_levels is not None and i in output_layer_levels:
                output[f"embedding_layer_{i}"] = out

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        return output


class Decoder_AE_MLP(BaseDecoder):
    def __init__(self, args: dict):
        super().__init__()
        self.input_dim = args.input_dim

        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(args.latent_dim, LAYER_1_SIZE), nn.Tanh()),
                nn.Sequential(nn.Linear(LAYER_1_SIZE, LAYER_1_SIZE), nn.Tanh()),
                nn.Linear(LAYER_1_SIZE, int(np.prod(args.input_dim))),
            ]
        )

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()
        out = z

        for i, layer in enumerate(self.layers, start=1):
            out = layer(out)
            if output_layer_levels is not None and i in output_layer_levels:
                output[f"reconstruction_layer_{i}"] = out

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)
        return output


class StreamlitLossCallback(TrainingCallback):
    """Update a Streamlit placeholder with latest training loss when logs are emitted."""

    def __init__(self, loss_placeholder, update_every: int = 100):
        super().__init__()
        self.loss_placeholder = loss_placeholder
        self.best_train_loss = float("inf")
        self.step = 0
        self.update_every = update_every

    def on_log(self, training_config, logs, **kwargs):
        self.step += 1
        loss = logs.get("train_epoch_loss")
        if loss is None:
            return

        if loss < self.best_train_loss:
            self.best_train_loss = loss

        if self.step % self.update_every == 0:
            global_step = kwargs.get("global_step", self.step)
            self.loss_placeholder.markdown(
                f"**Step** = {global_step}. **training loss** = {loss:.4f}. "
                f"**Best training loss** = {self.best_train_loss:.4f}"
            )


def train_pythae_vae(
    X: np.ndarray,
    latent_dim: int = 2,
    num_epochs: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    use_ae: bool = True,
    loss_callback: Optional[TrainingCallback] = None,
):
    """
    Train a Pythae AE/VAE on tabular data X (n_samples x n_features).

    Returns:
        trained_model, Z, X_recon_original_space, scaler
    """
    X = X.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_features = X.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_ae:
        model_config = AEConfig(input_dim=(n_features,), latent_dim=latent_dim)
        model = AE(
            model_config,
            encoder=Encoder_AE_MLP(model_config),
            decoder=Decoder_AE_MLP(model_config),
        ).to(device)
    else:
        model_config = VAEConfig(input_dim=(n_features,), latent_dim=latent_dim)
        model = VAE(
            model_config,
            encoder=Encoder_VAE_MLP(model_config),
            decoder=Decoder_AE_MLP(model_config),
        ).to(device)

    train_config = BaseTrainerConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=min(batch_size, len(X_scaled)),
        per_device_eval_batch_size=min(batch_size, len(X_scaled)),
        steps_saving=None,
        steps_predict=None,
        no_cuda=(device == "cpu"),
        keep_best_on_train=True,
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 2000, "factor": 0.9},
        optimizer_cls="RMSprop",
    )

    pipeline = TrainingPipeline(model=model, training_config=train_config)
    callbacks = [loss_callback] if loss_callback is not None else []
    pipeline(train_data=X_scaled, callbacks=callbacks)

    trained_model = pipeline.model.to(device)
    trained_model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        out = trained_model.predict(X_tensor)
        X_recon = out.recon_x.detach().cpu().numpy()
        Z = out.embedding.detach().cpu().numpy()

    return trained_model, Z, scaler.inverse_transform(X_recon), scaler


def make_vae_latent_and_manifold_figures(
    model,
    Z: np.ndarray,
    X: np.ndarray,
    X_recon_vae: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
    selected_features: list[str],
    grid_points: int = 15,
):
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
                hovertext=[f"Point #{i}<br>z1 = {Z[i,0]:.4f}" for i in range(Z.shape[0])],
            )
        )
        fig_latent.update_layout(
            xaxis_title="Sample index",
            yaxis_title="z1",
            title="latent space (1D)",
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
            title="latent space (2D)",
            height=500,
        )

    # ---------- Manifold & projection errors in original space ----------
    X3_orig = X[:, idxs]
    X3_recon = X_recon_vae[:, idxs]

    fig_manifold = go.Figure()

    fig_manifold.add_trace(
        go.Scatter3d(
            x=X3_orig[:, 0],
            y=X3_orig[:, 1],
            z=X3_orig[:, 2],
            mode="markers",
            marker=dict(size=6, color=ORIG_COLOR),
            name="Original",
            hoverinfo="text",
            hovertext=[
                f"Point #{i}<br>"
                + "<br>".join(f"{selected_features[j]} = {X3_orig[i, j]:.4f}" for j in range(3))
                for i in range(X3_orig.shape[0])
            ],
        )
    )

    fig_manifold.add_trace(
        go.Scatter3d(
            x=X3_recon[:, 0],
            y=X3_recon[:, 1],
            z=X3_recon[:, 2],
            mode="markers",
            marker=dict(size=4, symbol="x", color=RECON_COLOR),
            name="recon points",
            hoverinfo="text",
            hovertext=[
                f"Decoded point #{i}<br>"
                + "<br>".join(f"{selected_features[j]} = {X3_recon[i, j]:.4f}" for j in range(3))
                for i in range(X3_recon.shape[0])
            ],
        )
    )

    for i in range(X3_orig.shape[0]):
        fig_manifold.add_trace(
            go.Scatter3d(
                x=[X3_orig[i, 0], X3_recon[i, 0]],
                y=[X3_orig[i, 1], X3_recon[i, 1]],
                z=[X3_orig[i, 2], X3_recon[i, 2]],
                mode="lines",
                line=dict(width=2, color=ERROR_COLOR),
                showlegend=(i == 0),
                name="reconstruction error",
                hoverinfo="skip",
            )
        )

    # Nonlinear manifold from latent grid (black)
    if latent_dim == 1:
        z1_min, z1_max = Z[:, 0].min(), Z[:, 0].max()
        if z1_min == z1_max:
            z1_min -= 1.0
            z1_max += 1.0
        grid_z = np.linspace(z1_min, z1_max, grid_points).reshape(-1, 1)

        with torch.no_grad():
            z_tensor = torch.tensor(grid_z.astype(np.float32), device=device)
            decoded_out = model.decoder(z_tensor)
            decoded_unscaled = scaler.inverse_transform(decoded_out["reconstruction"].cpu().numpy())

        decoded3 = decoded_unscaled[:, idxs]
        fig_manifold.add_trace(
            go.Scatter3d(
                x=decoded3[:, 0],
                y=decoded3[:, 1],
                z=decoded3[:, 2],
                mode="lines",
                line=dict(width=3, color="black"),
                name="Decoded latent curve",
                hoverinfo="text",
                hovertext="Nonlinear manifold: decoded curve<br>Decoded from a z1 grid in latent space.",
            )
        )
    else:
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
            z_tensor = torch.tensor(grid_z.astype(np.float32), device=device)
            decoded_out = model.decoder(z_tensor)
            decoded = scaler.inverse_transform(decoded_out["reconstruction"].cpu().numpy())

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
                hovertext="Nonlinear manifold: decoded grid<br>Decoded from a regular (z1, z2) grid.",
                surfacecolor=np.zeros_like(decoded3[:, :, 0]),
                colorscale=[[0, "black"], [1, "black"]],
            )
        )

    fig_manifold.update_layout(
        scene=dict(
            xaxis_title=selected_features[0],
            yaxis_title=selected_features[1],
            zaxis_title=selected_features[2],
            aspectmode="data",
        ),
        title="Nonlinear manifold & reconstruction (original space)",
        height=750,
    )

    return fig_latent, fig_manifold


# ============================================================
# Streamlit app helpers
# ============================================================

def _ensure_table_shape(df: pd.DataFrame, num_features: int) -> pd.DataFrame:
    """Ensure df has exactly num_features columns named x1..xN, preserving overlapping columns."""
    new_cols = [f"x{i+1}" for i in range(num_features)]
    if df.shape[1] == num_features and list(df.columns) == new_cols:
        return df

    new_df = pd.DataFrame(index=df.index)
    for c in new_cols:
        if c in df.columns:
            new_df[c] = df[c]
        else:
            new_df[c] = np.random.randn(len(df))
    return new_df


def _parse_numeric_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        return df.astype(float).values
    except Exception:
        return None


# ============================================================
# Streamlit app
# ============================================================

def main():
    st.set_page_config(page_title="PCA & Auto-encoder Playground (Pythae)", layout="wide")
    st.title("PCA + Auto-encoder (Pythae) Interactive Playground")

    with st.expander("What this tool shows (for students / viewers)", expanded=False):
        st.markdown(
            """
- **PCA view**: linear projection onto principal components.
  - Original vs reconstructed points in original feature space.
  - **Error segments** from each point to its reconstruction.
  - A **projection line** (PC1) or **projection plane** (PC1–PC2).
- **VAE view (Pythae)**: nonlinear embedding.
  - Learns a latent space \\((z_1, z_2)\\).
  - Decode a **regular grid** in latent space to show a curved manifold in original coordinates.
            """
        )

    with st.sidebar:
        st.header("Settings")

        uploaded_file = st.file_uploader("Upload CSV (optional, numeric columns only)", type=["csv"])

        if "num_features" not in st.session_state:
            st.session_state.num_features = 3

        if uploaded_file is not None:
            try:
                df_csv = pd.read_csv(uploaded_file)
                df_num = df_csv.select_dtypes(include="number")
                if df_num.shape[1] == 0:
                    st.warning("Uploaded CSV has no numeric columns; keeping existing table.")
                else:
                    df_num = df_num.iloc[:, :10] if df_num.shape[1] > 10 else df_num
                    st.session_state.data_df = df_num.copy()
                    st.session_state.num_features = df_num.shape[1]
            except Exception as e:
                st.error(f"Could not read CSV: {e}")


        # NEW: number of rows (datapoints)
        if "num_rows" not in st.session_state:
            st.session_state.num_rows = 6
        
        st.text_input(
            "Number of datapoints (rows)",
            value=str(st.session_state.num_rows),
            key="num_rows_text",
            help="Enter an integer >= 1",
        )

        st.slider("Number of features (dimensions)", 1, 10, key="num_features")
        pcs_to_show = st.slider("Number of PCs to visualize", 1, 3, 2)
        show_scaled_space = st.checkbox("Show PCA geometry in scaled space", value=False)

        st.markdown("---")
        use_vae = st.checkbox("Enable Nonlinear Auto-encoding", value=True)
        use_ae = st.radio("Auto-encoder Type", ["AE", "VAE"]) == "AE"

        vae_epochs = 200
        latent_dim = 2
        if use_vae:
            vae_epochs = st.slider("Training epochs", 500, 50000, 2500, step=500)
            latent_dim = st.radio("latent dimension", [1, 2], index=1)
            st.caption("Latent dimension is fixed at 2 for 2D manifold visualization.")

    st.subheader("1. Enter / edit your data points")
    st.markdown(
        "Each **row** is a datapoint, each **column** is a feature. "
        "You can add or delete rows. All values are treated as numeric."
    )

    
    
    num_features = st.session_state.num_features
    
    # NEW: target number of rows from textbox
    target_rows = _coerce_num_rows(default=st.session_state.get("num_rows", 6), min_value=1)
    st.session_state.num_rows = target_rows  # keep a clean int copy
    
    if "data_df" not in st.session_state:
        st.session_state.data_df = pd.DataFrame(
            np.random.randn(target_rows, num_features),
            columns=[f"x{i+1}" for i in range(num_features)],
        )
    else:
        # keep columns in sync (existing helper)
        st.session_state.data_df = _ensure_table_shape(st.session_state.data_df, num_features)
    
        # NEW: resize rows while preserving existing values
        df = st.session_state.data_df
        if len(df) < target_rows:
            extra = pd.DataFrame(
                np.random.randn(target_rows - len(df), df.shape[1]),
                columns=df.columns,
            )
            st.session_state.data_df = pd.concat([df, extra], ignore_index=True)
        elif len(df) > target_rows:
            st.session_state.data_df = df.iloc[:target_rows].reset_index(drop=True)
    
        

    edited_df = st.data_editor(
        st.session_state.data_df,
        key="data_editor",
        num_rows="dynamic",
        use_container_width=True,
    )
    st.session_state.data_df = edited_df

    feature_names = list(edited_df.columns)
    num_features = len(feature_names)

    if "run_pca" not in st.session_state:
        st.session_state.run_pca = False

    if st.button("Run", type="primary"):
        st.session_state.run_pca = True

    if not st.session_state.run_pca:
        return

    X = _parse_numeric_matrix(edited_df[feature_names])
    if X is None:
        st.error("Could not parse table data as numeric.")
        st.session_state.run_pca = False
        return

    if X.shape[0] < 2:
        st.warning("Need at least 2 data points for PCA.")
        st.session_state.run_pca = False
        return

    # ====================================================
    # PCA
    # ====================================================
    pca, scaler, scores, X_recon = run_pca(X, pcs_to_show)

    st.subheader("2a. Reconstructed values and PCA scores")

    recon_df = pd.DataFrame(X_recon, columns=feature_names)
    st.markdown("**Reconstructed values (in original feature units):**")
    st.dataframe(recon_df.style.format("{:.4f}"), use_container_width=True)

    score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    st.markdown("**PCA scores (projection coordinates in PC-space):**")
    st.dataframe(scores_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("2b. PCA components and explained variance")

    comp_labels = [f"PC{i+1}" for i in range(pca.components_.shape[0])]
    loadings = pd.DataFrame(pca.components_, index=comp_labels, columns=feature_names)

    c1, c2 = st.columns((1.5, 1))
    with c1:
        st.markdown("**PCA loadings (components expressed in feature space):**")
        st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)

    with c2:
        evr_df = pd.DataFrame({"PC": comp_labels, "Explained variance ratio": pca.explained_variance_ratio_})
        st.markdown("**Explained variance ratio:**")
        st.dataframe(evr_df.style.format({"Explained variance ratio": "{:.3f}"}))

    st.subheader("3. PCA projection in original feature space")

    if num_features < 3:
        st.info(
            "You need at least 3 features to show a 3D plot. "
            "Increase the number of features in the sidebar."
        )
        return

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

    # ====================================================
    # AE / VAE nonlinear projection + manifold
    # ====================================================
    if not use_vae:
        return

    st.subheader("5. Nonlinear embedding (Pythae)")

    training_status = st.empty()
    with st.spinner("Training a small auto-encoder on your data..."):
        vae_callback = StreamlitLossCallback(training_status)
        model, Z, X_recon_vae, vae_scaler = train_pythae_vae(
            X,
            latent_dim=latent_dim,
            num_epochs=vae_epochs,
            learning_rate=1e-4,
            use_ae=use_ae,
            loss_callback=vae_callback,
        )

    selected_features_vae = st.multiselect(
        "Choose 3 features for the manifold in original space:",
        feature_names,
        default=feature_names[:3],
        key="vae_axes",
    )
    if len(selected_features_vae) != 3:
        st.warning("Please select exactly 3 features for the manifold.")
        return

    fig_latent, fig_manifold = make_vae_latent_and_manifold_figures(
        model,
        Z,
        X,
        X_recon_vae,
        vae_scaler,
        feature_names,
        selected_features_vae,
        grid_points=50,
    )
    st.plotly_chart(fig_latent, use_container_width=True)
    st.plotly_chart(fig_manifold, use_container_width=True)


if __name__ == "__main__":
    main()
