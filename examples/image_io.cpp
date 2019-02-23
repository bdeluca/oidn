// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <cmath>
#include <algorithm>
#include <fstream>
#include "image_io.h"

#ifdef HAS_OPEN_EXR
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#endif

namespace oidn {

#ifdef HAS_OPEN_EXR
  Imf::FrameBuffer frameBufferForEXR(const Tensor& image)
  {
    Imf::FrameBuffer frameBuffer;
    int xStride = image.dims[2]*sizeof(float);
    int yStride = image.dims[1]*image.dims[2]*sizeof(float);
    frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, (char*)&image[0], xStride, yStride));
    frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, (char*)&image[1], xStride, yStride));
    frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, (char*)&image[2], xStride, yStride));
    return frameBuffer;
  }

  Tensor loadImageEXR(const std::string& filename)
  {
    Imf::InputFile inputFile(filename.c_str());
    if (!inputFile.header().channels().findChannel("R") ||
        !inputFile.header().channels().findChannel("G") ||
        !inputFile.header().channels().findChannel("B"))
      throw std::invalid_argument("image must have 3 channels");
    Imath::Box2i dataWindow = inputFile.header().dataWindow();
    Tensor image({dataWindow.max.y-dataWindow.min.y+1, dataWindow.max.x-dataWindow.min.x+1, 3}, "hwc");
    inputFile.setFrameBuffer(frameBufferForEXR(image));
    inputFile.readPixels(dataWindow.min.y, dataWindow.max.y);
    return image;
  }

  void saveImageEXR(const Tensor& image, const std::string& filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");
    Imf::Header header(image.dims[1], image.dims[0], 1, Imath::V2f(0, 0), image.dims[1], Imf::INCREASING_Y, Imf::ZIP_COMPRESSION);
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
    Imf::OutputFile outputFile(filename.c_str(), header);
    outputFile.setFrameBuffer(frameBufferForEXR(image));
    outputFile.writePixels(image.dims[0]);
  }
#endif

  Tensor loadImagePFM(const std::string& filename)
  {
    // Open the file
    std::ifstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file '" + filename + "'");

    // Read the header
    std::string id;
    file >> id;
    int C;
    if (id == "PF")
      C = 3;
    else if (id == "Pf")
      C = 1;
    else
      throw std::runtime_error("invalid PFM image");

    int H, W;
    file >> W >> H;

    float scale;
    file >> scale;

    file.get(); // skip newline

    if (file.fail())
      throw std::runtime_error("invalid PFM image");

    if (scale >= 0.f)
      throw std::runtime_error("big-endian PFM images are not supported");
    scale = fabs(scale);

    // Read the pixels
    Tensor image({H, W, C}, "hwc");

    for (int h = 0; h < H; ++h)
    {
      for (int w = 0; w < W; ++w)
      {
        for (int c = 0; c < C; ++c)
        {
          float x;
          file.read((char*)&x, sizeof(float));
          image[((H-1-h)*W + w) * C + c] = x * scale;
        }
      }
    }

    if (file.fail())
      throw std::runtime_error("invalid PFM image");

    return image;
  }

  void saveImagePFM(const Tensor& image, const std::string& filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");
    const int H = image.dims[0];
    const int W = image.dims[1];
    const int C = image.dims[2];

    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file: '" + filename + "'");

    // Write the header
    file << "PF" << std::endl;
    file << W << " " << H << std::endl;
    file << "-1.0" << std::endl;

    // Write the pixels
    for (int h = 0; h < H; ++h)
    {
      for (int w = 0; w < W; ++w)
      {
        for (int c = 0; c < 3; ++c)
        {
          const float x = image[((H-1-h)*W + w) * C + c];
          file.write((char*)&x, sizeof(float));
        }
      }
    }
  }

  void saveImagePPM(const Tensor& image, const std::string& filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");
    const int H = image.dims[0];
    const int W = image.dims[1];
    const int C = image.dims[2];

    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file: '" + filename + "'");

    // Write the header
    file << "P6" << std::endl;
    file << W << " " << H << std::endl;
    file << "255" << std::endl;

    // Write the pixels
    for (int i = 0; i < W*H; ++i)
    {
      for (int c = 0; c < 3; ++c)
      {
        float x = image[i*C+c];
        x = pow(x, 1.f/2.2f);
        int ch = std::min(std::max(int(x * 255.f), 0), 255);
        file.put(char(ch));
      }
    }
  }

  std::string fileExtensionOf(const std::string& filename)
  {
    size_t index = filename.rfind('.');
    if (index == std::string::npos)
      throw std::invalid_argument("filename has no extension");
    return filename.substr(index+1);
  }

  Tensor loadImage(const std::string& filename)
  {
    std::string format = fileExtensionOf(filename);
#ifdef HAS_OPEN_EXR
    if (format == "exr")
      return loadImageEXR(filename);
    else
#endif
    if (format == "pfm")
      return loadImagePFM(filename);
    else
      throw std::invalid_argument("image format is not supported");
  }

  void saveImage(const Tensor& image, const std::string& filename)
  {
    std::string format = fileExtensionOf(filename);
#ifdef HAS_OPEN_EXR
    if (format == "exr")
      saveImageEXR(image, filename);
    else
#endif
    if (format == "pfm")
      saveImagePPM(image, filename);
    else
      throw std::invalid_argument("image format is not supported");
  }

} // namespace oidn
