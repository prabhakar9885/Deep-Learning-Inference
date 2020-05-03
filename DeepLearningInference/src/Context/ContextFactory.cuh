
#ifndef ContextFactory_CUH
#define ContextFactory_CUH

#include "ContextObject.cuh"

enum class ContextType
{
    cuBLAS,
    cuDNN
};

class ContextFactory
{
private:
    ContextObject contextObject;
public:
    ContextFactory();
    void createContext( ContextType contextType);
    ContextObject getContext();
    void releaseContext(ContextType contextType);
};


#endif // !ContextFactory_CUH