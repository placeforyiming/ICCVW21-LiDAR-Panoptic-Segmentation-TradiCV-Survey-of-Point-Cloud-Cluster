
#include <opencv/highgui.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>
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




class Euclidean_Cluster{
    //default is private
    public:
    
    double toler=0.02; // angle threshold of two points
    int min_size=0;  // The data of KITTI is not the original lidar signal, we need to consider those holes when search for nearest point in four directions.
	int max_size=1000000;
	//Eigen::MatrixXf position_e = Eigen::MatrixXf(3,3);
    
    


    Euclidean_Cluster(double toler, int min_size, int max_size):toler(toler),min_size(min_size),max_size(max_size){}
    
    

    std::array<int, 131072>  Euclidean_cluster(py::array_t<double> input_array_x_,py::array_t<double> input_array_y_, py::array_t<double> input_array_z_, py::array_t<double> mask_, int num_points){
    	// use np.reshape(-1) to strip the 2D image to 1d array
    	std::array<int, 131072> label_instance=std::array<int, 131072>();
    	auto input_array_x = input_array_x_.request();
    	double *ptr_x = (double *) input_array_x.ptr;
    	auto input_array_y = input_array_y_.request();
    	double *ptr_y = (double *) input_array_y.ptr;
		auto input_array_z = input_array_z_.request();
		double *ptr_z = (double *) input_array_z.ptr;
		auto mask = mask_.request();
		double *ptr_mask = (double *) mask.ptr;

    	pcl::PointCloud<pcl::PointXYZ> cloud;
    	
    	cloud.width=num_points;
    	cloud.height=1;
    	cloud.is_dense=false;
    	cloud.points.resize(cloud.width*cloud.height);
    	std::vector<int> index_original;
    	int count_points=0;
    	for (int i=0; i<131072;i++){
    		if (ptr_mask[i]>0){
    			cloud.points[count_points].x=ptr_x[i];
    			cloud.points[count_points].y=ptr_y[i];
    			cloud.points[count_points].z=ptr_z[i];
    			index_original.push_back(i);
    			count_points+=1; 
    		}


    	}


    	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    	*cloud_ptr= cloud; 

		
		  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
		  
		  tree->setInputCloud(cloud_ptr);

		  std::vector<pcl::PointIndices> cluster_indices;
		  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		  ec.setClusterTolerance(this->toler); // 2cm
		  ec.setMinClusterSize(this->min_size);
		  ec.setMaxClusterSize(this->max_size);
		  ec.setSearchMethod(tree);
		  ec.setInputCloud(cloud_ptr);
		  ec.extract(cluster_indices);


		  int j = 0;
		  int cluster_index=1;
		  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
		  {
		    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		    for (const auto& idx : it->indices) label_instance[index_original[idx]]=cluster_index;

		    j++;
			cluster_index++;
		  }

		  return label_instance;
}

};





PYBIND11_MODULE(Euclidean_Cluster, m) {
    py::class_<Euclidean_Cluster>(m, "Euclidean_Cluster")
    	.def(py::init<float, int, int >())
        .def("Euclidean_cluster", &Euclidean_Cluster::Euclidean_cluster);

}

