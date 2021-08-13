
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/point_cloud.h>

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

class ScanLineRun_Cluster{
    // define the key parameters of algorithm
public:
    float th_point2run = 0.5;  // Point to Run threshold
    float th_run2run = 1;     // Run to Run (merge) threshold

    ScanLineRun_Cluster(float th_point2run, float th_run2run): th_point2run(th_point2run), th_run2run(th_run2run){}

    // main function
    std::array<std::array<int, 2048 >,64 > ScanLineRun_cluster(py::array_t<double> input_array_x_,py::array_t<double> input_array_y_, py::array_t<double> input_array_z_, py::array_t<bool> mask_, int width, int height){
        std::array<std::array<int, 2048>, 64 > label_instance;
        label_instance.fill({});
        std::array<std::array<bool, 2048>, 64 > mask;
        pcl::PointCloud<pcl::PointXYZL> cloud;
        std::vector<std::vector<int>> rangeimage_Location_;

        std::vector<std::vector<int>> frame_idx_ (width);               // 2D vector stores the point idx over scanline structure 
        std::vector<std::vector<std::vector<int>>> runs_idx_ (width);   // 3D vector stores the point idx of runs over scanline structure
        std::vector<std::vector<int>> label_idx_ ;              // 2D struecture stores the point idx over different labels

        std::vector<pcl::PointXYZL> runsCurrent_;        // 2D structure to store the current runs
        std::vector<pcl::PointXYZL> runsAbove_;          // 2D structure to store the above runs
        std::vector<pcl::PointXYZL> runsAboveAbove_;     // 2D structure to store the runs above the above runs
        std::vector<pcl::PointXYZL> runsAboveAboveAbove_;
        std::vector<pcl::PointXYZL> runsAboveAboveAboveAbove_;

        // extract the data from py np array
        auto input_array_x = input_array_x_.request();
    	double *ptr_x = (double *) input_array_x.ptr;
    	auto input_array_y = input_array_y_.request();
    	double *ptr_y = (double *) input_array_y.ptr;
		auto input_array_z = input_array_z_.request();
		double *ptr_z = (double *) input_array_z.ptr;
        auto mask_valid = mask_.request();
        bool *ptr_mask_valid = (bool *) mask_valid.ptr;

        // store the data to Orgnized PointCloud, build Valid point Mask
        cloud.clear(); // need to clear the memory at cloud
        cloud.width = width;
        cloud.height = height;
        cloud.is_dense = false;
        

        
        int numPoint = 0;

        // ********************************************************
        // Create the Cloud and build frame index
        // ********************************************************

         for (int i = 0; i< width; i++)
        {
            for (int j = 0; j< height; j++)
            {
                mask[i][j] = ptr_mask_valid[j+i*height];
                if( mask[i][j] )  {
                    pcl::PointXYZL point;
                    point.x = ptr_x[j+i*height];
                    point.y = ptr_y[j+i*height];
                    point.z = ptr_z[j+i*height];
                    cloud.push_back(point);
                    frame_idx_[i].push_back(numPoint);
                    rangeimage_Location_.push_back({i,j});
                    numPoint ++;
                }
            }
        }

        // ********************************************************
        // Build the runs
        // ********************************************************
        
        for (int i = 0; i< width; i++)
        {
            int numPoints_row = frame_idx_[i].size();
            if (frame_idx_[i].size() == 0)
            {
                continue;
            }
            int maxRun = 0;                                        // intialize the run number of a single scanline
            runs_idx_[i].push_back(std::vector<int>());            // initialize a new run for the non-zero point scan line
            for (int j = 0; j<numPoints_row ; j++)
            {
                if (j != numPoints_row - 1)
                {
                    pcl::PointXYZL currentPoint = cloud.points[frame_idx_[i][j]];
                    pcl::PointXYZL nextPoint = cloud.points[frame_idx_[i][j+1]];
                    float dist = sqrt((currentPoint.x - nextPoint.x)*(currentPoint.x - nextPoint.x) + (currentPoint.y - nextPoint.y) * (currentPoint.y - nextPoint.y) + (currentPoint.z - nextPoint.z) * (currentPoint.z - nextPoint.z) );
                    runs_idx_[i][maxRun].push_back(frame_idx_[i][j]);
                    if (dist < th_point2run) 
                    {                          
                        continue;
                    }
                    else
                    {
                        runs_idx_[i].push_back(std::vector<int> ()); // add a new run
                        maxRun ++;
                    }
                }
                else // For the last point, we need to consider it as a circle
                {
                    pcl::PointXYZL currentPoint = cloud.points[frame_idx_[i][j]];
                    pcl::PointXYZL nextPoint = cloud.points[frame_idx_[i][0]];                    
                    float dist = sqrt((currentPoint.x - nextPoint.x)*(currentPoint.x - nextPoint.x) + (currentPoint.y - nextPoint.y) * (currentPoint.y - nextPoint.y) + (currentPoint.z - nextPoint.z) * (currentPoint.z - nextPoint.z) );
                    runs_idx_[i][maxRun].push_back(frame_idx_[i][j]);
                
                    if (dist < th_point2run && maxRun!=0) 
                    {                          
                        runs_idx_[i][0].insert(end(runs_idx_[i][0]), begin(runs_idx_[i][maxRun]), end(runs_idx_[i][maxRun])); // merge the last and first run when it connect to each other
                        runs_idx_[i].pop_back();                                                                              // when merge, get rid off the last run
                    }
                    else
                    {
                        continue;
                    }                          
                }
            }
        }

        // ********************************************************
        // Update the Label
        // ********************************************************
        for (int i = 0; i<width; i++)
        {
            if(frame_idx_[i].size() == 0 ){
                continue;
            }

            // find current run
            for (int j = 0; j<runs_idx_[i].size(); j++)
            {
                for (int k = 0; k<runs_idx_[i][j].size(); k++)
                {
                    runsCurrent_.push_back(cloud.points[runs_idx_[i][j][k]]);
                }
            }

            // First case: for the first non-empty line, intialize the label_idx_ and runsAbove
            if (label_idx_.empty()) 
            {
                int Point_idx = 0;
                for (int j = 0; j<runs_idx_[i].size(); j++)
                {
                    // Update the label index
                    label_idx_.push_back(runs_idx_[i][j]);

                    // Update the current Runs label
                    for (int k = 0; k<runs_idx_[i][j].size(); k++)
                    {
                        runsCurrent_[Point_idx].label = label_idx_.size() - 1;
                        Point_idx ++;        
                    }
                }
                runsAbove_ = runsCurrent_;
                runsCurrent_.clear();
                continue;
            }

            // Other cases: comparing the runsCurrent and runsAbove, get the labels to merge, update the currentAbove
            int point_idx_start = 0;            
            for (int j = 0; j<runs_idx_[i].size(); j++)
            {
                // For every run in runsCurrent, find label to merge
                std::vector<int> labelsToMerge;
                for (int k = 0; k<runs_idx_[i][j].size(); k++)
                {
                    int neighbor_idx = floor( (float) (point_idx_start+k) / runsCurrent_.size() * runsAbove_.size() );

                    pcl::PointXYZL neighborPoint = runsAbove_[neighbor_idx];
                    pcl::PointXYZL currentPoint = runsCurrent_[point_idx_start+k];
                    float dist_r = sqrt((currentPoint.x - neighborPoint.x)*(currentPoint.x - neighborPoint.x) + (currentPoint.y - neighborPoint.y) * (currentPoint.y - neighborPoint.y) + (currentPoint.z - neighborPoint.z) * (currentPoint.z - neighborPoint.z) ); 
                    
                    if(dist_r < th_run2run)
                    {
                        int label = neighborPoint.label;
                        labelsToMerge.push_back(label);
                    }
                }
                
                // Update the label index 
                // add the runs to a new label if labelsToMerge is empty  
                int runlabel;
                if (labelsToMerge.empty())
                {
                    // ************** make-up search ************** //
                    // bruteforce every point in above
                    for (int k = 0; k<runs_idx_[i][j].size(); k++) {
                        for (auto & abovePoint :  runsAbove_) {
                            pcl::PointXYZL currentPoint = runsCurrent_[point_idx_start+k];
                            float dist_r = sqrt((currentPoint.x - abovePoint.x)*(currentPoint.x - abovePoint.x) + (currentPoint.y - abovePoint.y) * (currentPoint.y - abovePoint.y) + (currentPoint.z - abovePoint.z) * (currentPoint.z - abovePoint.z) ); 
                            if(dist_r < th_run2run)
                            {
                                int label = abovePoint.label;
                                labelsToMerge.push_back(label);
                            }                            
                        }
                    }

                    // if still not bruteforce every point in the second above the current line
                    if (labelsToMerge.empty()){
                        for (int k = 0; k<runs_idx_[i][j].size(); k++) {
                            for (auto & abovePoint :  runsAboveAbove_) {
                                pcl::PointXYZL currentPoint = runsCurrent_[point_idx_start+k];
                                float dist_r = sqrt((currentPoint.x - abovePoint.x)*(currentPoint.x - abovePoint.x) + (currentPoint.y - abovePoint.y) * (currentPoint.y - abovePoint.y) + (currentPoint.z - abovePoint.z) * (currentPoint.z - abovePoint.z) ); 
                                if(dist_r < th_run2run)
                                {
                                    int label = abovePoint.label;
                                    labelsToMerge.push_back(label);
                                }                            
                            }
                        }
                    }

                    // if still not bruteforce every point in the third above the current line
                    if (labelsToMerge.empty()){
                        for (int k = 0; k<runs_idx_[i][j].size(); k++) {
                            for (auto & abovePoint :  runsAboveAboveAbove_) {
                                pcl::PointXYZL currentPoint = runsCurrent_[point_idx_start+k];
                                float dist_r = sqrt((currentPoint.x - abovePoint.x)*(currentPoint.x - abovePoint.x) + (currentPoint.y - abovePoint.y) * (currentPoint.y - abovePoint.y) + (currentPoint.z - abovePoint.z) * (currentPoint.z - abovePoint.z) ); 
                                if(dist_r < th_run2run)
                                {
                                    int label = abovePoint.label;
                                    labelsToMerge.push_back(label);
                                }                            
                            }
                        }
                    }

                    // if still not bruteforce every point in the third above the current line
                    if (labelsToMerge.empty()){
                        for (int k = 0; k<runs_idx_[i][j].size(); k++) {
                            for (auto & abovePoint :  runsAboveAboveAboveAbove_) {
                                pcl::PointXYZL currentPoint = runsCurrent_[point_idx_start+k];
                                float dist_r = sqrt((currentPoint.x - abovePoint.x)*(currentPoint.x - abovePoint.x) + (currentPoint.y - abovePoint.y) * (currentPoint.y - abovePoint.y) + (currentPoint.z - abovePoint.z) * (currentPoint.z - abovePoint.z) ); 
                                if(dist_r < th_run2run)
                                {
                                    int label = abovePoint.label;
                                    labelsToMerge.push_back(label);
                                }                            
                            }
                        }
                    }

                    if (!labelsToMerge.empty())
                    {
                        goto GotLabel;
                    }

                    // if still empty, generate a new label
                    label_idx_.push_back(runs_idx_[i][j]);
                    runlabel = label_idx_.size() - 1;                   
                }

                GotLabel:
                if (!labelsToMerge.empty())
                {   
                    // Sort and remove the duplicate labels
                    std::sort(labelsToMerge.begin(), labelsToMerge.end());
                    std::vector<int>::iterator it = std::unique(labelsToMerge.begin(), labelsToMerge.end());
                    labelsToMerge.erase(it, labelsToMerge.end());    
                                
                    // update label index: add the runs to target label
                    label_idx_[labelsToMerge[0]].insert(label_idx_[labelsToMerge[0]].end(), runs_idx_[i][j].begin(), runs_idx_[i][j].end());
                    // merge the labels
                    for (int label = 1; label < labelsToMerge.size(); label++ )
                    {
                        if(!label_idx_[labelsToMerge[label]].empty()){
                            label_idx_[labelsToMerge[0]].insert(label_idx_[labelsToMerge[0]].end(), label_idx_[labelsToMerge[label]].begin(), label_idx_[labelsToMerge[label]].end() );
                            int offset = labelsToMerge[label];
                            label_idx_[labelsToMerge[label]].clear(); // will cause some null label, but its okay
                        }
                    }
                    runlabel = labelsToMerge[0];
                }
                // Update the current Run label
                for (int k = 0; k<runs_idx_[i][j].size(); k++){
                    runsCurrent_[point_idx_start + k].label = runlabel;
                }

                // Update the start point index
                point_idx_start += runs_idx_[i][j].size();              
            }
            
            // // check the lines
            // std::cout   << "Current Line is " << i << std::endl
            //             << "-2: " << runsAboveAbove_.size() << std::endl
            //             << "-1: " << runsAbove_.size() << std::endl
            //             << " 0: " << runsCurrent_.size() << std::endl;                            
        
            // update the Above runs
            runsAboveAboveAboveAbove_ = runsAboveAboveAbove_; 
            runsAboveAboveAbove_ = runsAboveAbove_; 
            runsAboveAbove_ = runsAbove_; 
            runsAbove_ = runsCurrent_;
            runsCurrent_.clear();

        }

        // output labels
        int point_label = 1;
        for (auto & label:label_idx_){
            //std::cout << std::endl;
            for (auto & idx : label){
                int I = rangeimage_Location_[idx][0];
                int J = rangeimage_Location_[idx][1];
                label_instance[I][J] = point_label;
            }
            point_label++;
        }

        return label_instance;
    }
    
};


PYBIND11_MODULE(ScanLineRun_Cluster, m) {
    py::class_<ScanLineRun_Cluster>(m, "ScanLineRun_Cluster")
    	.def(py::init<float, float>())
        .def("ScanLineRun_cluster", &ScanLineRun_Cluster::ScanLineRun_cluster);
}