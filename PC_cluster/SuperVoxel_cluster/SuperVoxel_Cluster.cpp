#include <opencv/highgui.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <iostream>
#include <typeinfo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <typeinfo>
#include <cmath>
#include <algorithm>
#include <queue>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;


// define the object of each pixel position on range image


class SuperVoxel_Cluster{
        //default is private
    public:
    float voxel_resolution = 0.2;
    float seed_resolution = 0.25;
    float color_importance = 0.0f;
    float spatial_importance = 0.4f;
    float normal_importance = 1.0f;

    SuperVoxel_Cluster(float voxel_resolution, float seed_resolution, float color_importance, float spatial_importance, float normal_importance):voxel_resolution(voxel_resolution),seed_resolution(seed_resolution),color_importance(color_importance), spatial_importance(spatial_importance), normal_importance(normal_importance){} 

    std::array<std::array<int, 2048 >,64 > SuperVoxel_cluster(py::array_t<double> input_array_x_,py::array_t<double> input_array_y_, py::array_t<double> input_array_z_, int width, int height){
       // Load the data
        std::array<std::array<int, 2048>, 64 > label_instance;
    	auto input_array_x = input_array_x_.request();
    	double *ptr_x = (double *) input_array_x.ptr;
    	auto input_array_y = input_array_y_.request();
    	double *ptr_y = (double *) input_array_y.ptr;
		auto input_array_z = input_array_z_.request();
		double *ptr_z = (double *) input_array_z.ptr;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB> cloud; // Load PointCLoud
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_lable_ptr (new pcl::PointCloud<pcl::PointXYZL>); // Labeled Point Cloud Ptr
        
        // Read in the Orgnized Point Cloud
        cloud.width = width;
        cloud.height = height;
        cloud.is_dense = false;
        cloud.resize(width * height);
        

        for (int i = 0; i< cloud.width; i++)
        {
            for (int j =0; j< cloud.height; j++)
            {
                if( ptr_x[j+i*height] != -1 && ptr_y[j+i*height] != -1 && ptr_z[j+i*height] != -1 )  {
                    cloud.at(i,j).x = ptr_x[j+i*height];
                    cloud.at(i,j).y = ptr_y[j+i*height];
                    cloud.at(i,j).z = ptr_z[j+i*height];
                }
            }
        }

        *cloud_ptr = cloud;

        // Implement Voxel Clustering
        pcl::SupervoxelClustering<pcl::PointXYZRGB> super (voxel_resolution, seed_resolution);
            super.setUseSingleCameraTransform (false);
            super.setInputCloud (cloud_ptr);
            super.setColorImportance (color_importance);
            super.setSpatialImportance (spatial_importance);
            super.setNormalImportance (normal_importance);
            
            std::map <std::uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr > supervoxel_clusters;
            super.extract (supervoxel_clusters);

        cloud_lable_ptr = super.getLabeledCloud();
        auto cloudL = *cloud_lable_ptr;       

        for (int i = 0; i<cloudL.width; i++) {
            for (int j =0; j<cloudL.height; j++) {
                    label_instance[i][j] = cloudL.at(i,j).label;

                    if (cloudL.at(i,j).x == -1)
                    {
                        label_instance[i][j] = 0;
                    }
                }           
        }
         

        return label_instance;
        
    }

};

PYBIND11_MODULE(SuperVoxel_Cluster, m) {
    py::class_<SuperVoxel_Cluster>(m, "SuperVoxel_Cluster")
    	.def(py::init<float, float, float, float,float>())
        .def("SuperVoxel_cluster", &SuperVoxel_Cluster::SuperVoxel_cluster);

}
