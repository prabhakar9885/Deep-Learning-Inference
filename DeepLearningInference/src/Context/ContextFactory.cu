#include "./ContextFactory.cuh"

ContextFactory::ContextFactory()
{
}


void ContextFactory::createContext(ContextType contextType)
{
    switch (contextType)
    {
    case ContextType::cuBLAS:
        this->contextObject.releaseCublasHandle();
        this->contextObject.getCublasHandle();
        break;
    case ContextType::cuDNN:
        this->contextObject.releaseCudnnHandle();
        this->contextObject.getCudnnHandle();
        break;
    default:
        throw "Unsupported Context type is passed.";
    }
}


ContextObject ContextFactory::getContext()
{
    return this->contextObject;
}

void ContextFactory::releaseContext(ContextType contextType)
{
    switch (contextType)
    {
    case ContextType::cuBLAS:
        this->contextObject.releaseCublasHandle();
        break;
    case ContextType::cuDNN:
        this->contextObject.releaseCudnnHandle();
        break;
    default:
        throw "Unsupported Context type is passed.";
    }
}