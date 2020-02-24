#include <iostream>

#include "test_fixture.hpp"

using namespace srrg_hbst;

int main(int argc_, char** argv_) {
  testing::InitGoogleTest(&argc_, argv_);
  return RUN_ALL_TESTS();
}

TEST_F(HBST, SearchIdentical) {
  // ds populate the database
  Tree database;
  for (Tree::MatchableVector& matchables_train : matchables_train_per_image) {
    database.add(matchables_train, SplittingStrategy::SplitEven);
  }
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds query database with identical matchables
  for (size_t i = 0; i < 10; ++i) {
    const Tree::MatchableVector& matchables_query = matchables_train_per_image[i];
    Tree::MatchVectorMap matches;
    database.match(matchables_query, matches, 1);
    ASSERT_EQ(matches.size(), static_cast<size_t>(10));
    ASSERT_EQ(matches[i].size(), matchables_query.size());
    for (size_t j = 0; j < matches[i].size(); ++j) {
      ASSERT_EQ(matches[i][j].distance, 0);
    }
  }

  // ds clear database
  database.clear(true);
  ASSERT_EQ(database.size(), static_cast<size_t>(0));
}

TEST_F(HBST, SearchNoisy) {
  number_of_bits_to_flip = 10;

  // ds populate the database
  Tree database;
  for (Tree::MatchableVector& matchables_train : matchables_train_per_image) {
    database.add(matchables_train, SplittingStrategy::SplitEven);
  }
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds modify trained matchables slightly by flipping 10 arbitrary bits
  freeMatchablesQuery();
  matchables_query_per_image.reserve(matchables_train_per_image.size());
  for (Tree::MatchableVector& matchables_train : matchables_train_per_image) {
    Tree::MatchableVector matchables_query;
    matchables_query.reserve(matchables_train.size());
    for (Tree::Matchable* matchable_train : matchables_train) {
      const size_t image_identifier = matchable_train->objects.begin()->first;
      const size_t index_descriptor = matchable_train->objects.begin()->second;
      Tree::Descriptor descriptor   = matchable_train->descriptor;

      // ds create a noisy descriptor
      flipBits(descriptor);

      // ds add matchable to queries
      matchables_query.emplace_back(
        new Tree::Matchable(index_descriptor, descriptor, image_identifier));
    }
    matchables_query_per_image.emplace_back(matchables_query);
  }

  // ds query database with identical matchables
  for (size_t i = 0; i < 10; ++i) {
    const Tree::MatchableVector& matchables_query = matchables_query_per_image[i];
    Tree::MatchVectorMap matches;
    database.match(matchables_query, matches, 10);

    // ds we loose matches due to invalid partitioning
    ASSERT_GT(matches[i].size(), static_cast<size_t>(250));
    ASSERT_LT(matches[i].size(), static_cast<size_t>(400));

    // ds count correct matches
    size_t number_of_correct_matches = 0;
    for (size_t j = 0; j < matches[i].size(); ++j) {
      Tree::Match& match(matches[i][j]);
      ASSERT_LT(match.distance, 10);

      // ds indices match by construction
      if (match.object_query == match.object_references.front()) {
        ++number_of_correct_matches;
      }
    }
    ASSERT_GT(number_of_correct_matches, static_cast<size_t>(250));
  }

  // ds clear database
  database.clear(true);
  ASSERT_EQ(database.size(), static_cast<size_t>(0));
}
