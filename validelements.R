valid.elements <- function(col)
{
    n.na <- sum(is.na(col))
    if(n.na != 0)
    {
        return(1 - (n.na/length(col)))
    }
    n.empty <- sum(col == "")
    n.notnumber <- sum(col == "#DIV/0!")
    return(1 - (n.na+n.empty+n.notnumber)/length(col))
}
