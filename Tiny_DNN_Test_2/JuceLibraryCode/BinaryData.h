/* =========================================================================================

   This is an auto-generated file: Any edits you make may be overwritten!

*/

#pragma once

namespace BinaryData
{
    extern const char*   caffe_proto;
    const int            caffe_protoSize = 56728;

    extern const char*   CPPLINT_cfg;
    const int            CPPLINT_cfgSize = 25;

    extern const char*   ADSR_Int_Encoded_csv;
    const int            ADSR_Int_Encoded_csvSize = 781;

    extern const char*   Prelim_ADSR_CSV_txt;
    const int            Prelim_ADSR_CSV_txtSize = 2075;

    // Number of elements in the namedResourceList and originalFileNames arrays.
    const int namedResourceListSize = 4;

    // Points to the start of a list of resource names.
    extern const char* namedResourceList[];

    // Points to the start of a list of resource filenames.
    extern const char* originalFilenames[];

    // If you provide the name of one of the binary resource variables above, this function will
    // return the corresponding data and its size (or a null pointer if the name isn't found).
    const char* getNamedResource (const char* resourceNameUTF8, int& dataSizeInBytes);

    // If you provide the name of one of the binary resource variables above, this function will
    // return the corresponding original, non-mangled filename (or a null pointer if the name isn't found).
    const char* getNamedResourceOriginalFilename (const char* resourceNameUTF8);
}
