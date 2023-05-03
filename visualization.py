import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ipdb
import scipy
import scipy.spatial
from skimage import io, feature, color, transform

import plotly
import plotly.offline
import plotly.graph_objs as go


def get_plane(x, y, z):
    A = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    u,s,v = np.linalg.svd(A)
    return v[-1]

def run_plane_ransac(x, y, z, ignore_pts=None):
    tol = 1e-2
    n_iter = 10000
    max_inliers = 0
    data = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    if ignore_pts is None:
        ignore_pts = np.zeros((x.shape[0])).astype('bool')
        idx_to_choose = np.arange(x.shape[0])
    else:
        idx_to_choose = np.where(ignore_pts==0)[0]
    for i in range(n_iter):
        chosen_idx = np.random.choice(idx_to_choose, 4, replace=False)
        x_rand, y_rand, z_rand = x[chosen_idx], y[chosen_idx], z[chosen_idx]
        predicted_plane = get_plane(x_rand, y_rand, z_rand)
        predicted_plane_distance = (data * predicted_plane).sum(axis=1) / np.sqrt(np.square(predicted_plane[:-1]).sum())
        inliers = np.abs(predicted_plane_distance) < tol
        inliers[ignore_pts] = False
        inliers_count = inliers.sum()
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_inliers = inliers
    p = np.where(best_inliers==True)
    x_best, y_best, z_best = x[p], y[p], z[p]
    best_plane = get_plane(x_best, y_best, z_best)
    return best_plane, best_inliers

def refineF(F, pts1, pts2):
    def _singularize(F):
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        F = U.dot(np.diag(S).dot(V))
        return F

    def _objective_F(f, pts1, pts2):
        F = _singularize(f.reshape([3, 3]))
        num_points = pts1.shape[0]
        hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
        hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
        Fp1 = F.dot(hpts1.T)
        FTp2 = F.T.dot(hpts2.T)

        r = 0
        for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
            r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
        return r
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
        disp=False
    )
    return _singularize(f.reshape([3, 3]))

def compute_essential_matrix(F, K1, K2):
    return K2.T.dot(F).dot(K1)

def triangulate(C1, pts1, C2, pts2):
    projected_pts = []
    for i in range(len(pts1)):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A = np.array([x1 * C1[2].T - C1[0].T, y1 * C1[2].T - C1[1].T, x2 * C2[2].T - C2[0].T, y2 * C2[2].T - C2[1].T])
        U, S, V = np.linalg.svd(A)
        ppts = V[-1]
        ppts = ppts / ppts[-1]
        projected_pts.append(ppts)
    projected_pts = np.asarray(projected_pts)
    projected_pts1 = C1.dot(projected_pts.T)
    projected_pts2 = C2.dot(projected_pts.T)
    projected_pts1 = projected_pts1 / projected_pts1[-1,:]
    projected_pts2 = projected_pts2 / projected_pts2[-1,:]
    projected_pts1 = projected_pts1[:2,:].T
    projected_pts2 = projected_pts2[:2,:].T
    error = np.linalg.norm(projected_pts1 - pts1, axis=-1)**2 + np.linalg.norm(projected_pts2 - pts2, axis=-1)**2
    error = error.sum()
    return projected_pts[:,:3], error

def visualize_3d(pts_3d):

    # x, y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.scatter(x, y, z, color='blue')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,2].tolist(),
            z=(-pts_3d[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='firebrick',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=0.8
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="All Planes")
        }, auto_open=True)

def visualize_3d_planes(pts_3d_plane_1, pts_3d_plane_2):

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d_plane_1[:,0].tolist(),
            y=pts_3d_plane_1[:,2].tolist(),
            z=(-pts_3d_plane_1[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='lightcoral',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=1.0
            )
        )
    )

    total_points_list.append(go.Scatter3d(
            x=pts_3d_plane_2[:,0].tolist(),
            y=pts_3d_plane_2[:,2].tolist(),
            z=(-pts_3d_plane_2[:,1]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                color='limegreen',
                line=dict(
                    color='black',
                    width=0.1
                ),
                opacity=1.0
            )
        )
    )

    plotly.offline.plot({
            "data": total_points_list,
            "layout": go.Layout(title="Plane Fitting")
        }, auto_open=True)

def visualize_3d_color(pts_3d, img, key_pts, pts_old=None):
    key_pts = np.rint(key_pts).astype('int')
    kp_x = np.clip(key_pts[:,0], 0, img.shape[0]-1)
    kp_y = np.clip(key_pts[:,1], 0, img.shape[1]-1)
    r = img[2]
    g = img[1]
    b = img[0]

    colors = np.array([r,g,b]).T / 255.0

    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # for p, c in zip(pts_3d, colors):
    #     ax.plot([p[0]], [p[1]], [p[2]], '.', color=(c[0], c[1], c[2]), markersize=8, alpha=0.5)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.close()

    total_points_list = []
    total_points_list.append(go.Scatter3d(
            x=pts_3d[:,0].tolist(),
            y=pts_3d[:,1].tolist(),
            z=(pts_3d[:,2]).tolist(),
            mode='markers',
            marker=dict(
                size=5,
                line=dict(
                    color=colors,
                    width=0.1
                ),
                opacity=1
            )
        )
    )

    # total_points_list.append(go.Scatter3d(
    #             x=pts_old[0].tolist(),
    #             y=pts_old[1].tolist(),
    #             z=(pts_old[2]).tolist(),
    #             mode='markers',
    #             marker=dict(
    #                 size=2,
    #                 line=dict(
    #                     color='rgba(100, 100, 100, 0.14)',
    #                     width=0.1
    #                 ),
    #                 opacity=1
    #             )
    #         )
    #     )

    plotly.offline.plot({
                "data": total_points_list,
                "layout": go.Layout(title="3D Reconstruction")
            }, auto_open=True)

def get_plane(x, y, z):
    A = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    u,s,v = np.linalg.svd(A)
    return v[-1]

def run_plane_ransac(x, y, z, ignore_pts=None):
    tol = 1e-2
    n_iter = 10000
    max_inliers = 0
    data = np.stack((x,y,z,np.ones(x.shape[0])), 1)
    if ignore_pts is None:
        ignore_pts = np.zeros((x.shape[0])).astype('bool')
        idx_to_choose = np.arange(x.shape[0])
    else:
        idx_to_choose = np.where(ignore_pts==0)[0]
    for i in range(n_iter):
        chosen_idx = np.random.choice(idx_to_choose, 4, replace=False)
        x_rand, y_rand, z_rand = x[chosen_idx], y[chosen_idx], z[chosen_idx]
        predicted_plane = get_plane(x_rand, y_rand, z_rand)
        predicted_plane_distance = (data * predicted_plane).sum(axis=1) / np.sqrt(np.square(predicted_plane[:-1]).sum())
        inliers = np.abs(predicted_plane_distance) < tol
        inliers[ignore_pts] = False
        inliers_count = inliers.sum()
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_inliers = inliers
    p = np.where(best_inliers==True)
    x_best, y_best, z_best = x[p], y[p], z[p]
    best_plane = get_plane(x_best, y_best, z_best)
    return best_plane, best_inliers

def visualize_plane_fitting(plane_list, x, y, z):
    X_final = []
    Y_final = []
    Z_final = []
    k1, k2, k3, k4 = -1.5,1.5,-0.5,1.5
    for plane in plane_list:
        a, b, c, d = plane
        p = np.linspace(k1,k2,20)
        q = np.linspace(k3,k4,20)
        X,Y = np.meshgrid(p,q)
        Z = (-d - a*X - b*Y) / c
        X_final.append(X)
        Y_final.append(Y)
        Z_final.append(Z)

    fig = plt.figure()
    ax = Axes3D(fig)
    for idx in range(len(plane_list)):
        ax.plot_surface(X_final[idx], Y_final[idx], Z_final[idx], alpha=0.2)
    ax.scatter(x, y, z, color='green')
    plt.show()
    plt.close()

def ransacF(pts1, pts2, M):
    def _sevenpoint(pts1, pts2, M):
        pts1 = pts1.astype('float64')
        pts2 = pts2.astype('float64')
        pts1 /= M
        pts2 /= M
        A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
        A = np.asarray(A).T
        U, S, V = np.linalg.svd(A)
        F1 = V[-1].reshape(3,3)
        F2 = V[-2].reshape(3,3)
        func = lambda x: np.linalg.det((x * F1) + ((1 - x) * F2))
        x0 = func(0)
        x1 = (2 * (func(1) - func(-1)) / 3.0) - ((func(2) - func(-2)) / 12.0)
        x2 = (0.5 * func(1)) + (0.5 * func(-1)) - func(0)
        x3 = func(1) - x0 - x1 - x2
        alphas = np.roots([x3,x2,x1,x0])
        alphas = np.real(alphas[np.isreal(alphas)])
        final_F = []
        for a in alphas:
            F = (a * F1) + ((1 - a) * F2)
            F = refineF(F, pts1, pts2)
            t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
            F = t.T.dot(F).dot(t)
            final_F.append(F)
        return final_F

    def _eightpoint(pts1, pts2, M):
        pts1 = pts1.astype('float64')
        pts2 = pts2.astype('float64')
        pts1 /= M
        pts2 /= M
        A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
        A = np.asarray(A).T
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3,3)
        U, S, V = np.linalg.svd(F)
        S[-1] = 0.0
        F = U.dot(np.diag(S)).dot(V)
        F = refineF(F, pts1, pts2)
        t = np.array([[1.0/M,0.0,0.0],[0.0,1.0/M,0.0],[0.0,0.0,1.0]])
        F = t.T.dot(F).dot(t)
        return F

    num_iter = 200
    threshold = 1e-3
    total = len(pts1)
    max_inliers = 0
    for i in range(num_iter):
        idx = np.random.permutation(np.arange(total))[:7]
        selected_pts1, selected_pts2 = pts1[idx], pts2[idx]
        F7 = _sevenpoint(selected_pts1, selected_pts2, M)
        for k, F in enumerate(F7):
            pts1_homo = np.concatenate((pts1, np.ones((pts1.shape[0],1))), axis=-1)
            pts2_homo = np.concatenate((pts2, np.ones((pts2.shape[0],1))), axis=-1)
            error = []
            for p, q in zip(pts1_homo, pts2_homo):
                error.append(q.T.dot(F).dot(p))
            error = np.abs(np.asarray(error))
            inliers = error < threshold

            if inliers.sum() > max_inliers:
                max_inliers = inliers.sum()
                best_inliers = inliers
                best_k = k
    selected_pts1, selected_pts2 = pts1[best_inliers], pts2[best_inliers]
    F = _eightpoint(selected_pts1, selected_pts2, M)
    return F

def get_3d_pts(E, pts1, pts2, K1, K2):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K1)

    M1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    M2 = np.hstack((R, t))

    C1 = np.dot(K1,  M1)
    C2 = np.dot(K2,  M2)
    P, _ = triangulate(C1, pts1, C2, pts2)
    val = 6
    flag1 = (P < -val)
    flag2 = (P > val)
    outliers = np.logical_or(flag1, flag2).any(axis=-1)
    P = P[~outliers]
    # outliers_ = P[:,2]<12
    # P = P[~outliers_]
    return P, C1, C2, [pts1, pts2]
