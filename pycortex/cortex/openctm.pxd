cdef extern from "openctm.h":
    ctypedef enum CTMenum:
        CTM_NONE
        CTM_INVALID_CONTEXT
        CTM_INVALID_ARGUMENT
        CTM_INVALID_OPERATION
        CTM_INVALID_MESH
        CTM_OUT_OF_MEMORY
        CTM_FILE_ERROR
        CTM_BAD_FORMAT
        CTM_LZMA_ERROR
        CTM_INTERNAL_ERROR
        CTM_UNSUPPORTED_FORMAT_VERSION
        CTM_IMPORT
        CTM_EXPORT
        CTM_METHOD_RAW
        CTM_METHOD_MG1
        CTM_METHOD_MG2
        CTM_VERTEX_COUNT
        CTM_TRIANGLE_COUNT
        CTM_HAS_NORMALS
        CTM_UV_MAP_COUNT
        CTM_ATTRIB_MAP_COUNT
        CTM_VERTEX_PRECISION
        CTM_NORMAL_PRECISION
        CTM_COMPRESSION_METHOD
        CTM_FILE_COMMENT
        CTM_NAME
        CTM_FILE_NAME
        CTM_PRECISION
        CTM_INDICES
        CTM_VERTICES
        CTM_NORMALS
        CTM_UV_MAP_1
        CTM_UV_MAP_2
        CTM_UV_MAP_3
        CTM_UV_MAP_4
        CTM_UV_MAP_5
        CTM_UV_MAP_6
        CTM_UV_MAP_7
        CTM_UV_MAP_8
        CTM_ATTRIB_MAP_1
        CTM_ATTRIB_MAP_2
        CTM_ATTRIB_MAP_3
        CTM_ATTRIB_MAP_4
        CTM_ATTRIB_MAP_5
        CTM_ATTRIB_MAP_6
        CTM_ATTRIB_MAP_7
        CTM_ATTRIB_MAP_8

    ctypedef void* CTMcontext

    CTMcontext ctmNewContext(CTMenum mode)
    void ctmFreeContext(CTMcontext context)
    CTMenum ctmGetError(CTMcontext context)
    char* ctmErrorString(CTMenum error)
    unsigned int ctmGetInteger(CTMcontext context, CTMenum property)
    float ctmGetFloat(CTMcontext context, CTMenum property)
    unsigned int* ctmGetIntegerArray(CTMcontext context, CTMenum property)
    float* ctmGetFloatArray(CTMcontext context, CTMenum property)
    CTMenum ctmGetNamedUVMap(CTMcontext context, char* name)
    char* ctmGetUVMapString(CTMcontext context, CTMenum uvMap, CTMenum stringType)
    float ctmGetUVMapFloat(CTMcontext context, CTMenum uvMap, CTMenum floatType)
    CTMenum ctmGetNamedAttribMap(CTMcontext context, char* name)
    char* ctmGetAttribMapString(CTMcontext context, CTMenum attribMap, CTMenum stringType)
    float ctmGetAttribMapFloat(CTMcontext context, CTMenum attribMap, CTMenum floatType)
    char* ctmGetString(CTMcontext context, CTMenum stringType)
    void ctmCompressionMethod(CTMcontext context, CTMenum method)
    void ctmCompressionLevel(CTMcontext context, unsigned int level)
    void ctmVertexPrecision(CTMcontext context, float precision)
    void ctmVertexPrecisionRel(CTMcontext context, float precision)
    void ctmNormalPrecision(CTMcontext context, float precision)
    void ctmUVCoordPrecision(CTMcontext context, CTMenum uvMap, float precision)
    void ctmAttribPrecision(CTMcontext context, CTMenum attribMap, float precision)
    void ctmFileComment(CTMcontext context, char* comment)
    void ctmDefineMesh(CTMcontext context, float* vertices, unsigned int vertexCount, unsigned int* indices, unsigned int indexCount, float* normals)
    CTMenum ctmAddUVMap(CTMcontext context, float* uv, char* name, char* fileName)
    CTMenum ctmAddAttribMap(CTMcontext context, float* attrib, char* name)
    void ctmLoad(CTMcontext context, char* fileName)
    void ctmSave(CTMcontext context, char* fileName)

