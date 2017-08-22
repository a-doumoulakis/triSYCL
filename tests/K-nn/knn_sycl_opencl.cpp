/* RUN: %{execute}%s

   Digit recognition in images using nearest neighbour matching
   with the SYCL OpenCL interoperability mode
*/

#include "util.hpp"

using namespace cl::sycl;

range<1> global_size {5000};

std::vector<Img> training_set;
std::vector<Img> validation_set;
int result[training_set_size];


int search_image(buffer<int>& training, buffer<int>& res,
                 const Img& img, queue& q, const kernel& k) {

  {
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    // Compute the L2 distance between an image and each one from the
    // training set
    q.submit([&] (handler &cgh) {
        // Set the kernel arguments. The accessors lazily trigger data
        // transfers between host and device only if necessary. For
        // example "training" and "res" are only transfered to the device
        // the first time the kernel is executed and "res" is transfered back
        // after every execution
        cgh.set_args(training.get_access<access::mode::read>(cgh),
                     A.get_access<access::mode::read>(cgh),
                     res.get_access<access::mode::discard_write>(cgh),
                     int { training_set_size }, int { pixel_number });
        // Launch the kernel with training_set_size work-items
        cgh.parallel_for(global_size, k);
      });
  }
  auto r = res.get_access<access::mode::read>();
  // Find the image with the minimum distance
  auto min_image = std::min_element(std::begin(result), std::end(result));

  // Test if we found the good digit
  return
    training_set[std::distance(std::begin(result), min_image)].label == img.label;
}

int test_main(int argc, char* argv[]) {
  training_set = slurp_file<Img>(KNN_TRAIN_DATA_PATH);
  validation_set =  slurp_file<Img>(KNN_VALID_DATA_PATH);
  buffer<int> training_buffer = get_buffer<Img>(training_set);
  buffer<int> result_buffer { result, training_set_size };

  // Device selection
  boost::compute::device device = boost::compute::system::default_device();
  std::cout << "\nUsing " << device.name() << std::endl;

  // Boost context and queue to allow us to choose
  // whichever device we want
  boost::compute::context context { device };
  boost::compute::command_queue b_queue { context, device };

  // A SYCL queue to send the heterogeneous work-load to
  queue q { b_queue };

  auto program = boost::compute::program::create_with_source(R"(
    __kernel void kernel_compute(__global const int* trainingSet,
                                 __global const int* data,
                                 __global int* res, int setSize, int dataSize) {
      int diff, toAdd, computeId;
      computeId = get_global_id(0);
      if (computeId < setSize) {
        diff = 0;
        for (int i = 0; i < dataSize; i++) {
            toAdd = data[i] - trainingSet[computeId*dataSize + i];
            diff += toAdd * toAdd;
        }
        res[computeId] = diff;
      }
    }
    )", context);

  program.build();

  // Construct a SYCL kernel from OpenCL kernel to be used in
  // interoperability mode
  kernel k { boost::compute::kernel { program, "kernel_compute"} };

  int correct = 0;

  // Match each image from the validation set against the images from
  // the training set
  for (auto const & img : validation_set)
    correct += search_image(training_buffer, result_buffer, img, q, k);

  BOOST_CHECK((100.0*correct/validation_set.size()) == 94.4);

  return 0;
}
