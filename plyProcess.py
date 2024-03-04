import open3d as o3d

def display_ply_point_cloud(ply_filename):
    # 读取PLY点云文件
    point_cloud = o3d.io.read_point_cloud(ply_filename)

    #显示点云
    o3d.visualization.draw_geometries([point_cloud])


ply_filename = 'test2.ply'
display_ply_point_cloud( ply_filename)