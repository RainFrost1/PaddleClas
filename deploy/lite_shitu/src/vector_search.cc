// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/vector_search.h"

#include <faiss/AuxIndexStructures.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>

namespace PPShiTu {
// load the vector.index
void VectorSearch::LoadIndexFile() {
  std::string file_path = this->index_dir + OS_PATH_SEP + "vector.index";
  const char *fname = file_path.c_str();
  this->index = faiss::read_index(fname, 0);
}

// load index_id2image_id.txt
void VectorSearch::LoadIdMap() {
  std::string file_path = this->index_dir + OS_PATH_SEP + "id_label_map.txt";
  std::ifstream in(file_path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      std::regex ws_re("\\s+");
      std::vector<std::string> v(
          std::sregex_token_iterator(line.begin(), line.end(), ws_re, -1),
          std::sregex_token_iterator());
      if (v.size() != 3) {
        std::cout << "The number of element for each line in : " << file_path
                  << "must be 3: index_id image_id label, exit the program..."
                  << std::endl;
        exit(1);
      } else {
        this->index_id2image_id.insert(std::pair<int64_t, std::string>(
            std::stol(v[0], nullptr, 10), v[1]));
        this->image_id2index_id.insert(std::pair<std::string, int64_t>(
            v[1], std::stol(v[0], nullptr, 10)));
        this->image_id2label.insert(
            std::pair<std::string, std::string>(v[1], v[2]));
      }
    }
  }
}

// doing search
const SearchResult &VectorSearch::Search(float *feature, int query_number) {
  this->D.resize(this->return_k * query_number);
  this->I.resize(this->return_k * query_number);
  this->index->search(query_number, feature, return_k, D.data(), I.data());
  this->sr.return_k = this->return_k;
  this->sr.D = this->D;
  this->sr.I = this->I;
  return this->sr;
}

const std::string &VectorSearch::GetLabel(faiss::Index::idx_t ind) {
  return this->image_id2label.at(this->index_id2image_id.at(ind));
}

const std::string &VectorSearch::GetImageID(faiss::Index::idx_t ind) {
  return this->index_id2image_id.at(ind);
}

int VectorSearch::AddFeature(float *feature, std::string image_id,
                             std::string label) {
  this->index->add(1, feature);
  int id = this->index_id2image_id.size();
  if (label != "") {
    this->index_id2image_id.insert(
        std::pair<int64_t, std::string>(id, image_id));
    this->image_id2index_id.insert(
        std::pair<std::string, int64_t>(image_id, id));
    this->image_id2label.insert(
        std::pair<std::string, std::string>(image_id, label));
  } else {
    this->index_id2image_id.insert(
        std::pair<int64_t, std::string>(id, image_id));
    this->image_id2index_id.insert(
        std::pair<std::string, int64_t>(image_id, id));
    this->image_id2label.insert(
        std::pair<std::string, std::string>(image_id, std::to_string(id)));
  }
  return this->index->ntotal;
}

void VectorSearch::SaveIndex(std::string save_dir) {
  std::string file_path_index, file_path_labelmap;
  if (save_dir == "") {
    file_path_index = this->index_dir + OS_PATH_SEP + "vector.index";
    file_path_labelmap = this->index_dir + OS_PATH_SEP + "id_label_map.txt";
  } else {
    file_path_index = save_dir + OS_PATH_SEP + "vector.index";
    file_path_labelmap = save_dir + OS_PATH_SEP + "id_label_map.txt";
  }
  // save index
  faiss::write_index(this->index, file_path_index.c_str());

  // save label_map
  std::ofstream out(file_path_labelmap);
  std::map<int64_t, std::string>::iterator iter;
  for (iter = this->index_id2image_id.begin();
       iter != this->index_id2image_id.end(); iter++) {
    std::string content = std::to_string(iter->first) + " " + iter->second +
                          " " + this->image_id2label.at(iter->second);
    out.write(content.c_str(), content.size());
    out << std::endl;
  }
  out.close();
  printf("save_index\n");
}

void VectorSearch::ClearFeature() {
  faiss::IDSelectorRange ids(0, this->index->ntotal);
  this->index->remove_ids(ids);
  // std::map<int64_t, std::string>::iterator iter;
  // iter = this->index_id2image_id.find(this->index_len);
  this->index_id2image_id.erase(this->index_id2image_id.begin(),
                                this->index_id2image_id.end());
  this->image_id2index_id.erase(this->image_id2index_id.begin(),
                                this->image_id2index_id.end());
}

bool VectorSearch::RemoveFeature(std::vector<std::string> image_ids) {
  std::vector<int64_t> ids;
  for (int i = 0; i < image_ids.size(); i++) {
    if (this->image_id2index_id.find(image_ids[i]) ==
        this->image_id2index_id.end())
      return false;
    ids.push_back(this->image_id2index_id[image_ids[i]]);
  }
  sort(ids.begin(), ids.end());
  // reverse(ids.begin(), ids.end());
  faiss::IDSelectorBatch batch_ids(ids.size(), ids.data());
  this->index->remove_ids(batch_ids);
  std::vector<std::pair<int64_t, int64_t>> id_changes;
  for (int64_t i = ids[0]; i < this->index->ntotal; i++) {
    int index_change = 0;
    bool flag = true;
    for (int j = 0; j < ids.size(); j++) {
      if (i == ids[j]) {
        flag = false;
        break;
      } else if (i > ids[j])
        index_change += 1;
      else
        break;
    }
    if (flag) {
      id_changes.push_back(std::pair<int64_t, int64_t>(i, i - index_change));
    }
  }

  for (int i = 0; i < id_changes.size(); i++) {
    this->index_id2image_id[id_changes[i].second] =
        this->index_id2image_id[id_changes[i].first];
    this->image_id2index_id[this->index_id2image_id[id_changes[i].second]] =
        id_changes[i].second;
  }
  for (int64_t i = this->index_id2image_id.size() - ids.size();
       i < this->index_id2image_id.size(); i++)
    this->index_id2image_id.erase(i);
  for (int i = 0; i < image_ids.size(); i++) {
    this->image_id2index_id.erase(image_ids[i]);
    this->image_id2label.erase(image_ids[i]);
  }
  return true;
}
}  // namespace PPShiTu
