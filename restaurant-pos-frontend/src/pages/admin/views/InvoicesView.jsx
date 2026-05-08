import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { Receipt, RefreshCw, Loader2 } from 'lucide-react'
import { fetchInvoices } from '../../../services/platformApi'

export default function InvoicesView() {
  const [invoices, setInvoices] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const data = await fetchInvoices()
      setInvoices(data)
    } catch (err) {
      toast.error(err.message || 'Failed to load invoices', { id: 'invoices-load-error' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const totalRevenue = invoices.reduce((s, inv) => s + (inv.total_cents || 0), 0)

  if (loading) return <div className="page-loading"><Loader2 className="spin" size={24} />Loading invoices...</div>

  return (
    <div style={{ padding: 16 }}>
      {/* Summary */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="stat-card" style={{ minWidth: 160 }}>
          <div className="stat-label">Total Invoices</div>
          <div className="stat-value" style={{ fontSize: 24 }}>{invoices.length}</div>
        </div>
        <div className="stat-card" style={{ minWidth: 160 }}>
          <div className="stat-label">Total Revenue</div>
          <div className="stat-value" style={{ fontSize: 24, color: 'var(--brand-green)' }}>
            ₹{(totalRevenue / 100).toLocaleString('en-IN', { minimumFractionDigits: 0 })}
          </div>
        </div>
        <button className="icon-btn" style={{ alignSelf: 'center', marginLeft: 'auto' }} onClick={load}>
          <RefreshCw size={15} />
        </button>
      </div>

      {invoices.length === 0 && (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 0', fontSize: 14 }}>
          No invoices yet. Invoices are generated when table sessions are closed.
        </div>
      )}

      {invoices.length > 0 && (
        <div className="orders-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Invoice No</th>
                <th>Type</th>
                <th>Entity</th>
                <th>Subtotal</th>
                <th>Tax</th>
                <th>Total</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {invoices.map(inv => (
                <tr key={inv.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Receipt size={14} style={{ color: 'var(--brand-primary)' }} />
                      <span style={{ fontFamily: 'monospace', fontWeight: 700, color: 'var(--brand-primary)' }}>
                        {inv.invoice_no}
                      </span>
                    </div>
                  </td>
                  <td>
                    <span style={{ fontSize: 11, background: 'var(--bg-overlay)', padding: '2px 8px', borderRadius: 6 }}>
                      {inv.entity_type || '—'}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>#{inv.entity_id}</td>
                  <td>₹{((inv.subtotal_cents || 0) / 100).toFixed(2)}</td>
                  <td style={{ color: 'var(--text-muted)' }}>₹{((inv.tax_cents || 0) / 100).toFixed(2)}</td>
                  <td style={{ fontWeight: 700, color: 'var(--brand-green)' }}>
                    ₹{((inv.total_cents || 0) / 100).toFixed(2)}
                  </td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {inv.created_at ? new Date(inv.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
