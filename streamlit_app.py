import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import torch
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainer, TrainerConfig
from pythae.data.datasets import NumpyDataset

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


def make_3d_pca_original_space_figure(
    X, X_recon, pca, scaler, feature_names, pcs_to_show, selected_features
):
    """
    3D plot in ORIGINAL feature space:
      - Original points
      - Reconstructed points
      - Reconstruction error lines
      - Projection line (1 PC) or plane (2 PCs)
      - Rich hovertext for teaching
    """
    idxs = [feature_names.index(f) for f in selected_features]
    X3 = X[:, idxs]
    X3_recon = X_recon[:, idxs]

    n = X3.shape[0]
    # L2 reconstruction error per point (in full space, but show value)
    full_err = np.linalg.norm(X - X_recon, axis=1)

    fig = go.Figure()

    # Hovertext helpers
    def point_hover(i, kind):
        coords = "<br>".join(
            f"{selected_features[j]} = {X3[i, j]:.3f}" for j in range(3)
        )
        rcoords = "<br>".join(
            f"{selected_features[j]} = {X3_recon[i, j]:.3f}" for j in range(3)
        )
        return (
            f"{kind} point #{i}<br>"
            + "Original (subset):<br>"
            + coords
            + "<br><br>Reconstructed (subset):<br>"
            + rcoords
            + f"<br><br>Total reconstruction error (L2, full {X.shape[1]}D) = {full_err[i]:.4f}"
        )

    # Original points
    fig.add_trace(
        go.Scatter3d(
            x=X3[:, 0],
            y=X3[:, 1],
            z=X3[:, 2],
            mode="markers",
            marker=dict(size=6),
            name="Original",
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
            marker=dict(size=4, symbol="x"),
            name="Reconstructed",
            hovertext=[point_hover(i, "Reconstructed") for i in range(n)],
            hoverinfo="text",
        )
    )

    # Reconstruction error lines
    for i in range(n):
        fig.add_trace(
            go.Scatter3d(
                x=[X3[i, 0], X3_recon[i, 0]],
                y=[X3[i, 1], X3_recon[i, 1]],
                z=[X3[i, 2], X3_recon[i, 2]],
                mode="lines",
                line=dict(width=2),
                showlegend=(i == 0),
                name="Reconstruction error",
                hoverinfo="text",
                hovertext=(
                    f"Error segment for point #{i}<br>"
                    f"L2 error (subset 3D) = {np.linalg.norm(X3[i] - X3_recon[i]):.4f}<br>"
                    f"L2 error (full {X.shape[1]}D) = {full_err[i]:.4f}"
                ),
            )
        )

    # Projection geometry: line or plane in original space
    X_scaled = scaler.transform(X)
    mean_scaled = X_scaled.mean(axis=0)
    scales = scaler.scale_

    if pcs_to_show == 1:
        pc1 = pca.components_[0]  # in scaled space
        direction_orig = pc1 / scales
        center_orig = scaler.inverse_transform(mean_scaled)[idxs]

        t = np.linspace(-3, 3, 30)
        line_points = np.array(
            [center_orig + alpha * direction_orig[idxs] for alpha in t]
        )

        fig.add_trace(
            go.Scatter3d(
                x=line_points[:, 0],
                y=line_points[:, 1],
                z=line_points[:, 2],
                mode="lines",
                line=dict(width=4),
                name="PC1 line (projection direction)",
                hoverinfo="text",
                hovertext="PC1 line in original space<br>"
                          "Each point is μ + t·v, where v is PC1 mapped back "
                          "to original feature units.",
            )
        )

    elif pcs_to_show == 2 and pca.components_.shape[0] >= 2:
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        v1 = pc1 / scales
        v2 = pc2 / scales
        center_orig = scaler.inverse_transform(mean_scaled)[idxs]

        grid_lin = np.linspace(-2, 2, 20)
        a, b = np.meshgrid(grid_lin, grid_lin)
        plane = (
            center_orig
            + a[..., None] * v1[idxs][None, None, :]
            + b[..., None] * v2[idxs][None, None, :]
        )

        fig.add_trace(
            go.Surface(
                x=plane[:, :, 0],
                y=plane[:, :, 1],
                z=plane[:, :, 2],
                opacity=0.35,
                showscale=False,
                name="PC1–PC2 plane",
                hoverinfo="text",
                hovertext="Projection plane spanned by PC1 & PC2<br>"
                          "Shown in original feature units.",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title=selected_features[0],
            yaxis_title=selected_features[1],
            zaxis_title=selected_features[2],
        ),
        title="Original space: PCA projection & reconstruction",
        height=750,
    )
    return fig


def make_pca_score_figure(scores, pcs_to_show):
    n_samples, n_components = scores.shape
    pcs_to_show = min(pcs_to_show, n_components)

    if pcs_to_show == 1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_samples),
                y=scores[:, 0],
                mode="markers+lines",
                name="PC1 score",
                hoverinfo="text",
                hovertext=[
                    f"Point #{i}<br>PC1 score = {scores[i,0]:.4f}"
                    for i in range(n_samples)
                ],
            )
        )
        fig.update_layout(
            xaxis_title="Sample index",
            yaxis_title="PC1 score",
            title="PCA scores (1D)",
            height=500,
        )

    elif pcs_to_show == 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode="markers",
                name="Scores",
                hoverinfo="text",
                hovertext=[
                    f"Point #{i}<br>PC1 = {scores[i,0]:.4f}<br>PC2 = {scores[i,1]:.4f}"
                    for i in range(n_samples)
                ],
            )
        )
        fig.update_layout(
            xaxis_title="PC1",
            yaxis_title="PC2",
            title="PCA scores (2D)",
            height=500,
        )

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=scores[:, 0],
                y=scores[:, 1],
                z=scores[:, 2],
                mode="markers",
                marker=dict(size=5),
                name="Scores",
                hoverinfo="text",
                hovertext=[
                    f"Point #{i}<br>"
                    f"PC1 = {scores[i,0]:.4f}<br>"
                    f"PC2 = {scores[i,1]:.4f}<br>"
                    f"PC3 = {scores[i,2]:.4f}"
                    for i in range(n_samples)
                ],
            )
        )
        fig.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            title="PCA scores (3D)",
            height=700,
        )

    return fig


# ============================================================
# Pythae VAE utilities
# ============================================================

def train_pythae_vae(X, latent_dim=2, epochs=200, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create VAE model
    model_config = VAEConfig(
        input_dim=(X.shape[1],),
        latent_dim=latent_dim,
        reconstruction_loss="mse",
    )
    model = VAE(model_config).to(device)

    # Dataset wrapper
    dataset = NumpyDataset(X.astype(np.float32))

    # Training config (REPLACES TrainerConfig)
    train_config = TrainingConfig(
        output_dir="vae_tmp",
        num_epochs=epochs,
        batch_size=min(batch_size, len(X)),
        learning_rate=1e-3,
    )
    
    trainer = BaseTrainer(
        model=model,
        train_dataset=NumpyDataset(X.astype(np.float32)),
        training_config=train_config,
    )

    trainer.train()

    # After training, compute reconstructions + latent
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X.astype(np.float32)).to(device)
        outputs = model(X_tensor)
        X_recon = outputs["reconstruction"].cpu().numpy()
        Z = outputs["z"].cpu().numpy()

    return model, Z, X_recon



def make_vae_latent_and_manifold_figures(
    model, Z, X_recon_vae, feature_names, selected_features, grid_points=15
):
    """
    - 2D latent scatter with hovertext.
    - Latent grid → decoded → 3D manifold surface in original feature coordinates.
    """
    device = next(model.parameters()).device
    idxs = [feature_names.index(f) for f in selected_features]

    # ---------- Latent scatter (assumes 2D) ----------
    fig_latent = go.Figure()
    fig_latent.add_trace(
        go.Scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            mode="markers",
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

    # ---------- Latent grid → decoded manifold ----------
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

    # 3D manifold surface + VAE recon points
    fig_manifold = go.Figure()

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
                "Each surface point comes from decoding a regular grid in latent (z1, z2)."
            ),
        )
    )

    fig_manifold.add_trace(
        go.Scatter3d(
            x=X_recon_vae[:, idxs[0]],
            y=X_recon_vae[:, idxs[1]],
            z=X_recon_vae[:, idxs[2]],
            mode="markers",
            marker=dict(size=4),
            name="VAE recon points",
            hoverinfo="text",
            hovertext=[
                f"Decoded point #{i}<br>"
                + "<br>".join(
                    f"{selected_features[j]} = {X_recon_vae[i, idxs[j]]:.4f}"
                    for j in range(3)
                )
                for i in range(X_recon_vae.shape[0])
            ],
        )
    )

    fig_manifold.update_layout(
        scene=dict(
            xaxis_title=selected_features[0],
            yaxis_title=selected_features[1],
            zaxis_title=selected_features[2],
        ),
        title="VAE nonlinear manifold (decoded latent grid in original space)",
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
        num_features = st.slider("Number of features (dimensions)", 2, 10, 3)
        pcs_to_show = st.slider("Number of PCs to visualize", 1, 3, 2)

        st.markdown("---")
        use_vae = st.checkbox("Enable VAE (Pythae nonlinear view)", value=True)
        vae_epochs = 200
        latent_dim = 2
        if use_vae:
            vae_epochs = st.slider("VAE training epochs", 50, 800, 250, step=50)
            st.caption("Latent dimension is fixed at 2 for 2D manifold visualization.")

    feature_names = [f"x{i+1}" for i in range(num_features)]

    st.subheader("1. Enter / edit your data points")
    st.markdown(
        "Each **row** is a datapoint, each **column** is a feature. "
        "You can add or delete rows. All values are treated as numeric."
    )

    default_rows = 6
    default_data = np.random.randn(default_rows, num_features)
    default_df = pd.DataFrame(default_data, columns=feature_names)

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    if st.button("Run PCA (and VAE if enabled)", type="primary"):
        # ---------- Prepare X ----------
        try:
            X = edited_df[feature_names].astype(float).values
        except Exception as e:
            st.error(f"Could not parse data as numeric: {e}")
            return

        n_samples = X.shape[0]
        if n_samples < 2:
            st.warning("Need at least 2 data points for PCA.")
            return

        # ====================================================
        # PCA
        # ====================================================
        pca, scaler, scores, X_recon = run_pca(X, pcs_to_show)

        st.subheader("2. PCA components and explained variance")

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
                fig_orig = make_3d_pca_original_space_figure(
                    X, X_recon, pca, scaler, feature_names, pcs_to_show, selected_features
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
        st.subheader("4. PCA score space (projection coordinates)")
        st.markdown(
            "Here we look at the coordinates of each point **in PC-space** rather than original feature space."
        )
        fig_scores = make_pca_score_figure(scores, pcs_to_show)
        st.plotly_chart(fig_scores, use_container_width=True)

        # ====================================================
        # VAE (Pythae) nonlinear projection + manifold
        # ====================================================
        if use_vae:
            st.subheader("5. VAE nonlinear embedding (Pythae)")

            with st.spinner("Training a small VAE on your data..."):
                model, Z, X_recon_vae = train_pythae_vae(
                    X,
                    latent_dim=latent_dim,
                    epochs=vae_epochs,
                    batch_size=16,
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
