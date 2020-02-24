#include <iostream>
#include <srrg_hbst/types/binary_tree.hpp>

// ds we associate our data with integer indexes (uint64_t)
typedef srrg_hbst::BinaryTree256<uint64_t> Tree;

// ds dummy descriptor generation
Tree::MatchableVector getDummyMatchables(const uint64_t& number_of_matchables_,
                                         const uint64_t& identifier_tree_);

int32_t main() {
  // ds obtain a vector of reference matchables
  std::vector<Tree::MatchableVector> matchables_per_image(10);
  for (uint32_t u = 0; u < matchables_per_image.size(); ++u) {
    matchables_per_image[u] = getDummyMatchables(10000, u);
  }

  // ds allocate a BTree object on these descriptors (no shared pointer passed as the tree will have
  // its own constant copy of the train descriptors)
  std::cerr << "building initial tree" << std::endl;
  Tree hbst_tree(0, matchables_per_image[0]);
  std::cerr << "built initial tree" << std::endl;

  // ds add other elements to the tree
  std::cerr << "growing tree" << std::endl;
  for (uint32_t u = 1; u < matchables_per_image.size(); ++u) {
    std::cerr << "training on matchables of image: " << u << std::endl;
    hbst_tree.add(matchables_per_image[u], srrg_hbst::SplittingStrategy::SplitEven);
  }
  std::cerr << "grown tree" << std::endl;

  // ds query descriptor pool
  const Tree::MatchableVector matchables_query =
    getDummyMatchables(5000, matchables_per_image.size() + 1);

  // ds query the tree with maximum distance 25 - and obtain individual matches for each image added
  std::cerr << "querying tree - getting results per image" << std::endl;
  Tree::MatchVectorMap matches_per_image;
  hbst_tree.match(matchables_query, matches_per_image, 25);
  std::cerr << "queried tree" << std::endl;

  // ds query the tree with maximum distance 25 - obtaining total matches over all descriptors added
  std::cerr << "querying tree - getting results for all images" << std::endl;
  Tree::MatchVector matches_overall;
  hbst_tree.match(matchables_query, matches_overall, 25);
  std::cerr << "queried tree" << std::endl;

  // ds clear query matchables (not stored in the tree)
  for (const Tree::Matchable* matchable : matchables_query) {
    delete matchable;
  }

  // ds clear matchables stored in the tree
  hbst_tree.clear(true);
  return 0;
}

Tree::MatchableVector getDummyMatchables(const uint64_t& number_of_matchables_,
                                         const uint64_t& identifier_tree_) {
  // ds preallocate vector
  Tree::MatchableVector matchables(number_of_matchables_);

  // ds set values
  for (uint64_t identifier = 0; identifier < number_of_matchables_; ++identifier) {
    // ds generate a "random" descriptor by flipping some bits
    Tree::Matchable::Descriptor descriptor;
    for (uint32_t u = 0; u < 256 /*descriptor size in bits*/ / 3; ++u) {
      if (rand() % 2) {
        descriptor.flip(u);
      }
    }

    // ds set matchable
    matchables[identifier] = new Tree::Matchable(identifier, descriptor, identifier_tree_);
  }

  // ds done
  return matchables;
}
