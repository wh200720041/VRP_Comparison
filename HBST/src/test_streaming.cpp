#include <fstream>
#include <iostream>

#include "test_fixture.hpp"

using namespace srrg_hbst;

int main(int argc_, char** argv_) {
  testing::InitGoogleTest(&argc_, argv_);
  return RUN_ALL_TESTS();
}

TEST_F(HBST, Write) {
  // ds populate the database
  Tree database;
  for (Tree::MatchableVector& matchables_train : matchables_train_per_image) {
    database.add(matchables_train, SplittingStrategy::SplitEven);
  }
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds verify matching before serializing
  for (Tree::MatchableVector& matchables_query : matchables_query_per_image) {
    Tree::MatchVectorMap match_vectors;
    database.match(matchables_query, match_vectors);
    ASSERT_EQ(match_vectors.size(), static_cast<size_t>(10));
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_query.size());
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_train.size());
    ASSERT_EQ(match_vectors.at(0).size(), matching_distances.size());
    for (size_t i = 0; i < match_vectors.at(0).size(); ++i) {
      // ds check match against hardcoded sampling ground truth
      const Tree::Match& match = match_vectors.at(0)[i];
      ASSERT_EQ(match.object_query, identifiers_query[i]);
      ASSERT_GE(match.object_references.size(), static_cast<size_t>(1));
      ASSERT_EQ(match.object_references[0], identifiers_train[i]);
      ASSERT_EQ(match.distance, matching_distances[i]);
    }
  }

  // ds save database to disk (this operation will not clean dynamic memory)
  ASSERT_TRUE(database.write("database.hbst"));

  // ds clear database
  database.clear(true);
}

TEST_F(HBST, Read) {
  // ds load database from disk
  Tree database;
  ASSERT_EQ(database.size(), static_cast<size_t>(0));
  ASSERT_TRUE(database.read("database.hbst"));
  ASSERT_EQ(database.size(), static_cast<size_t>(10));

  // ds verify matching after deserializing
  for (Tree::MatchableVector& matchables_query : matchables_query_per_image) {
    Tree::MatchVectorMap match_vectors;
    database.match(matchables_query, match_vectors);
    ASSERT_EQ(match_vectors.size(), static_cast<size_t>(10));
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_query.size());
    ASSERT_EQ(match_vectors.at(0).size(), identifiers_train.size());
    ASSERT_EQ(match_vectors.at(0).size(), matching_distances.size());
    for (size_t i = 0; i < match_vectors.at(0).size(); ++i) {
      // ds check match against hardcoded sampling ground truth
      const Tree::Match& match = match_vectors.at(0)[i];
      ASSERT_EQ(match.object_query, identifiers_query[i]);
      ASSERT_GE(match.object_references.size(), static_cast<size_t>(1));
      ASSERT_EQ(match.object_references[0], identifiers_train[i]);
      ASSERT_EQ(match.distance, matching_distances[i]);
    }
  }

  // ds clear database
  database.clear(true);
}
