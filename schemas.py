from pydantic import BaseModel, Field
from typing import List, Optional

class LineItem(BaseModel):
    item_name: str = Field(description="Nama barang atau produk")
    item_quantity: float = Field(description="Jumlah atau kuantitas barang")
    item_price: Optional[float] = Field(description="Harga satuan barang")
    item_total: float = Field(description="Total harga untuk item tersebut")

class ReceiptData(BaseModel):
    merchant_name: str = Field(description="Nama toko atau tempat usaha")
    merchant_address: Optional[str] = Field(description="Alamat toko jika ada")
    transaction_date: Optional[str] = Field(description="Tanggal transaksi dalam format YYYY-MM-DD")
    transaction_time: Optional[str] = Field(description="Waktu transaksi dalam format HH:MM:SS")
    total_amount: float = Field(description="Total keseluruhan pembayaran")
    line_items: List[LineItem] = Field(description="Daftar item yang dibeli")