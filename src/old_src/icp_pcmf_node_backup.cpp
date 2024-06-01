#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class ICPNode
{
public:
    ICPNode() : nh_("~"), transformation_matrix_(Eigen::Matrix4f::Identity())
    {
        // Subscribers
        map_points_sub_ = nh_.subscribe("/map_points_processed", 1, &ICPNode::mapPointsCallback, this);
        path_points_sub_ = nh_.subscribe("/hdl_graph_slam/path_points", 1, &ICPNode::pathPointsCallback, this);
        reference_map_sub_ = nh_.subscribe("/reference_map", 1, &ICPNode::referenceMapCallback, this);

        // Publishers for transformed map and path
        transformed_map_pub_ = nh_.advertise<PointCloud>("transformed_map", 1);
        transformed_path_pub_ = nh_.advertise<PointCloud>("transformed_path", 1);
    }

    void mapPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        if (!reference_map_.empty())
        {
            // Perform ICP if reference map is available
            PointCloud::Ptr transformed_cloud(new PointCloud);
            performICP(reference_map_, *cloud, *transformed_cloud);

            // Publish the transformed map
            transformed_map_pub_.publish(transformed_cloud);
        }
    }

    void referenceMapCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        pcl::fromROSMsg(*msg, reference_map_);
    }

    void pathPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        PointCloud::Ptr cloud(new PointCloud);
        pcl::fromROSMsg(*msg, *cloud);

        // Transform path points using the stored transformation matrix
        PointCloud::Ptr transformed_path(new PointCloud);
        pcl::transformPointCloud(*cloud, *transformed_path, transformation_matrix_);

        // Publish the transformed path
        transformed_path_pub_.publish(transformed_path);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_points_sub_;
    ros::Subscriber path_points_sub_;
    ros::Subscriber reference_map_sub_;
    ros::Publisher transformed_map_pub_;
    ros::Publisher transformed_path_pub_;

    PointCloud reference_map_;
    Eigen::Matrix4f transformation_matrix_;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

    void performICP(const PointCloud& reference, const PointCloud& input, PointCloud& output)
    {
        // // Align centroids of reference and input
        // Eigen::Vector4f reference_centroid;
        // pcl::compute3DCentroid(reference, reference_centroid);
        // Eigen::Vector4f input_centroid;
        // pcl::compute3DCentroid(input, input_centroid);
        
        // Eigen::Matrix4f translation_matrix = Eigen::Matrix4f::Identity();
        // translation_matrix.block<3,1>(0,3) = reference_centroid.head<3>() - input_centroid.head<3>();
        
        // PointCloud::Ptr input_transformed(new PointCloud);
        // pcl::transformPointCloud(input, *input_transformed, translation_matrix);

        // Perform ICP
        // icp_.setInputSource(input_transformed);
        icp_.setInputSource(input.makeShared());
        icp_.setInputTarget(reference.makeShared());
        icp_.align(output);

        // Store the final transformation matrix
        // transformation_matrix_ = translation_matrix * icp_.getFinalTransformation();
        transformation_matrix_ = icp_.getFinalTransformation();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "icp_pcmf_node");
    ICPNode icp_node;
    ros::spin();
    return 0;
}
