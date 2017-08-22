/* RUN: %{execute}%s

   Digit recognition in images using nearest neighbour matching.
   A full description as well as an OpenCL implementation and a
   ComputeCPP compatible version can be found at
   https://github.com/a-doumoulakis/triSYCL_knn/
*/

#include "util.hpp"

using namespace cl::sycl;

class KnnKernel;

std::vector<Img> training_set;
std::vector<Img> validation_set;
int result[training_set_size];


int search_image(buffer<int>& training, buffer<int>& res_buffer,
                 const Img& img, queue& q) {

  {
    buffer<int> A { std::begin(img.pixels), std::end(img.pixels) };
    // Compute the L2 distance between an image and each one from the
    // training set
    q.submit([&] (handler &cgh) {
        // These accessors lazily trigger data transfers between host
        // and device only if necessary. For example "training" is
        // only transfered the first time the kernel is executed.
        auto train = training.get_access<access::mode::read>(cgh);
        auto ka = A.get_access<access::mode::read>(cgh);
        auto kb = res_buffer.get_access<access::mode::write>(cgh);
        // Launch a kernel with training_set_size work-items
        cgh.parallel_for<class KnnKernel>(range<1> { training_set_size },
                                          [=] (id<1> index) {
            decltype(ka)::value_type diff = 0;
            // For each pixel
            for (auto i = 0; i != pixel_number; i++) {
              auto toAdd = ka[i] - train[index[0]*pixel_number + i];
              diff += toAdd*toAdd;
            }
            kb[index] = diff;
          });
      });
  }

  auto r = res_buffer.get_access<access::mode::read>();

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

  // A SYCL queue to send the heterogeneous work-load to
  queue q;

  int correct = 0;

  // Match each image from the validation set against the images from
  // the training set
  for (auto const& img : validation_set)
    correct += search_image(training_buffer, result_buffer, img, q);

  BOOST_CHECK((100.0*correct/validation_set.size()) == 94.4);

  return 0;
}
