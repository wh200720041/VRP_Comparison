#include <gtest/gtest.h>
#include <random>

#include "srrg_hbst/types/binary_tree.hpp"

typedef srrg_hbst::BinaryTree256<size_t> Tree;

// ds hbst test fixture
class HBST : public ::testing::Test {
protected:
  void SetUp() override {
    Tree::Node::maximum_partitioning = 0.45;            // ds noisy, synthetic case
    random_number_generator          = std::mt19937(0); // ds locked seed for reproducibility
    generateMatchables(matchables_train_per_image, number_of_images_train, 0);
    generateMatchables(matchables_query_per_image, number_of_images_query, number_of_images_train);
  }

  void TearDown() override {
    // ds training matchables are already freed by the tree
    freeMatchablesQuery();
  }

  void generateMatchables(std::vector<Tree::MatchableVector>& matchables_per_image_,
                          const size_t& number_of_images_,
                          const size_t& image_number_start_) {
    matchables_per_image_.clear();
    matchables_per_image_.reserve(number_of_images_);

    // ds populate matchable vectors
    for (size_t index_image = 0; index_image < number_of_images_; ++index_image) {
      Tree::MatchableVector matchables;
      matchables.reserve(number_of_matchables_per_image);
      for (size_t index_descriptor = 0; index_descriptor < number_of_matchables_per_image;
           ++index_descriptor) {
        Tree::Descriptor descriptor;
        ASSERT_EQ(descriptor.size(), Tree::Matchable::descriptor_size_bits);
        flipBits(descriptor);
        matchables.emplace_back(
          new Tree::Matchable(index_descriptor, descriptor, index_image + image_number_start_));
      }
      matchables_per_image_.emplace_back(matchables);
    }
  }

  void freeMatchablesQuery() {
    for (const Tree::MatchableVector& matchables : matchables_query_per_image) {
      for (const Tree::Matchable* matchable : matchables) {
        delete matchable;
      }
    }
    matchables_query_per_image.clear();
  }

  void flipBits(Tree::Descriptor& descriptor_) {
    for (size_t flips = 0; flips < number_of_bits_to_flip; ++flips) {
      std::uniform_int_distribution<uint32_t> bit_index_to_flip(
        0, Tree::Matchable::descriptor_size_bits - 1);
      descriptor_.set(bit_index_to_flip(random_number_generator));
    }
  }

  //! dummy query and reference (= database) matchables
  std::vector<Tree::MatchableVector> matchables_query_per_image;
  std::vector<Tree::MatchableVector> matchables_train_per_image;

  //! configuration
  static constexpr size_t number_of_images_query         = 1;
  static constexpr size_t number_of_images_train         = 10;
  static constexpr size_t number_of_matchables_per_image = 1000;

  //! random number generator used to generate binary descriptors
  static std::mt19937 random_number_generator;
  size_t number_of_bits_to_flip = 18;

  //! "ground truth" data (set in tests)
  static std::vector<size_t> identifiers_query;
  static std::vector<size_t> identifiers_train;
  static std::vector<Tree::real_type> matching_distances;
};

// ds come on c++11
constexpr size_t HBST::number_of_images_query;
constexpr size_t HBST::number_of_images_train;
constexpr size_t HBST::number_of_matchables_per_image;
std::mt19937 HBST::random_number_generator;

// ds streaming "ground truth" assuming consistent random sampling on testing architectures
std::vector<size_t> HBST::identifiers_query = {
  58,  78,  91,  108, 109, 116, 122, 127, 144, 146, 149, 150, 151, 159, 183, 187,
  190, 192, 200, 212, 217, 241, 247, 258, 259, 266, 309, 312, 320, 340, 349, 360,
  372, 377, 412, 421, 443, 449, 464, 466, 480, 492, 497, 506, 509, 573, 575, 586,
  598, 621, 626, 627, 641, 666, 680, 681, 705, 707, 737, 755, 763, 765, 766, 776,
  783, 815, 852, 855, 859, 881, 888, 895, 903, 907, 921, 951, 952, 955};
std::vector<size_t> HBST::identifiers_train = {
  512, 896, 874, 621, 495, 998, 289, 321, 965, 615, 658, 359, 251, 99,  965, 679,
  751, 270, 407, 471, 615, 110, 932, 193, 270, 719, 373, 939, 675, 474, 867, 648,
  342, 300, 811, 771, 182, 180, 464, 865, 275, 877, 707, 618, 366, 320, 577, 722,
  563, 235, 699, 507, 91,  37,  468, 52,  947, 559, 765, 427, 852, 125, 279, 754,
  392, 904, 807, 380, 311, 662, 839, 83,  212, 96,  546, 46,  354, 872};
std::vector<Tree::real_type> HBST::matching_distances = {
  22, 23, 24, 24, 24, 24, 24, 22, 24, 22, 24, 24, 24, 24, 23, 24, 24, 24, 24, 23,
  24, 24, 24, 23, 24, 23, 24, 23, 24, 24, 22, 23, 24, 23, 23, 23, 24, 24, 24, 24,
  24, 24, 23, 24, 24, 22, 24, 23, 23, 23, 23, 24, 24, 23, 24, 24, 20, 24, 23, 22,
  24, 20, 24, 20, 24, 22, 23, 24, 24, 23, 24, 24, 23, 24, 23, 24, 23, 24};
